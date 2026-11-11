# Dual AMN Upsert — Güncelleme kazanımı, bağımsız bilgileri koruma

Tarih: **11 Kasım 2026**  
Depo: **ulusluyz/AMN-4W-Edit**  
Eski arşiv: **AMN-4W-jules yalnızca okunur; değiştirilmedi**

## Sonuç: mevcut eğitilmiş Dual checkpoint'lerinde sıfır yeniden eğitim

Mevcut Dual AMN ağırlıkları hiçbir şekilde değiştirilmedi ve yeni model parametresi eklenmedi. Ayrı `DualAddressUpsertAMN` modeli aynı state_dict'i `strict=True` ile yükler.

**5 seed × mevcut checkpoint × tam aynı test örnekleri:**

| Görev | Eski Dual AMN | Upsert Dual AMN | Standart delta AMN |
|---|---:|---:|---:|
| Son yazılan bilginin geçerli olması | 1.256/2.500 (%50,24) | **2.500/2.500 (%100)** | 2.500/2.500 (%100) |
| 4 farklı anahtarı okuma | 15.000/15.000 | **15.000/15.000** | 14.996/15.000 |
| 6 farklı anahtarı okuma | 5.000/5.000 | **5.000/5.000** | 4.994/5.000 |
| 8 farklı anahtarı okuma | 5.000/5.000 | **5.000/5.000** | 4.990/5.000 |
| 12 farklı anahtarı okuma | 5.000/5.000 | **5.000/5.000** | 4.919/5.000 |

**Yeni Dual toplam 32.500/32.500 doğru.** Bu küçük, tanımlı **yapılandırılmış Türkçe ilişkisel** görevde gözlenen sonuçtur; genel doğal dil doğruluğu olarak yorumlanamaz.

Beş seed'in her birinde güncelleme başarısı: **500/500**. Eski Dual aynı örneklerde sırayla **250, 249, 262, 256, 239** doğru verdi.

## Düzeltilen matematik

Eski Dual yöntem bütün eski ve yeni kayıtları Gram sistemine birlikte sokuyordu. Aynı kişi ve ilişki iki kez yazıldığında iki anahtar **tam aynı** olduğundan doğrusal bağımlılık oluşuyordu. Eski değer ile yeni değer arasında doğru bir son-yazma tercihi yoktu.

Yeni çözüm, matrisi kurmadan önce aynı `(entity_id, relation_id)` ikilisinin **yalnızca son görülen kaydını** etkin bırakır. Değer vektörlerinin veya ilişki anahtarlarının eğitimde yeniden öğrenilmesi gerekmez.

```
keep[i] = not any(j > i and key[j] == key[i])
K_active = diag(keep) @ K
V_active = diag(keep) @ V
G = K_active @ K_active.T + 1e-5 I + diag(not keep)
D = solve(G, K_active)
S = V_active.T @ D
output = classifier(S @ query_key)
```

Atılmış anahtarların Gram çözümünü bozmaması için o satır/sütunlara birim diyagonal eklenir. Geri kalan aktif anahtarlar önceki dual sistemin korumalı ilişkisel okuma özelliğini sürdürür.

## Gerileme kontrolü

- **Güncelleme (upsert):** 2.500/2.500.
- **4/6/8/12 kayıtlı benzersiz anahtarlar:** Eski Dual ile aynı doğruluk, toplam 30.000/30.000.
- **Matematiksel ileri hesaplama:** Benzersiz anahtarlı tüm denemelerde `torch.equal` ile **tam eşit logit**; ölçülen maksimum fark **0**.
- **Gradyanlar:** Benzersiz anahtarlı regresyon testinde her parametre için birebir aynı.
- **Farklı batch örneklerinde değişik tekrar yerleşimleri ve aynı anahtarın üç kez yazılması:** regresyon testiyle kontrol edildi.
- **Eğitilmiş ağırlıklara müdahale:** yok; **yeniden eğitim süresi = 0**.
- **Hız sınırı:** Otomatik tekrar algılama ek çıkarım hesaplaması gerektirir. Örneğin benzersiz anahtarların **dışarıdan garanti edildiği** bir akışta `guarantee_unique_keys=True` kullanılması eski Dual `forward` yoluna doğrudan döner ve o hesaplama yükünü atlar. Bu seçeneği tekrar içeren bağlamda açmak yanlıştır.
- **Korumanın geçerlilik alanı:** Gözlenen yapılandırılmış görevde gerileme yok. Tüm olası veriler, kararsız/çakışık öğrenilmiş anahtarlar, çok daha uzun bellekler ve GPU hızına dair sıfır kayıp garantisi yoktur.

## Gerçek model çıktı örneği

**Bağlam:** Ayşe'nin kaleminin rengi sarı. Elif şu anda kafede. Sena'nın kaleminin rengi yeşil. **Ayşe'nin kaleminin rengi kırmızı.**

**Soru:** Ayşe'nin kaleminin rengi nedir?

- Standart AMN: **kırmızı**.
- Eski Dual AMN: **sarı** — yanlış.
- **Dual Upsert: kırmızı** — doğru.

Bu Türkçe metinler ortak kural tabanlı ayrıştırıcıyla yapılandırılmış kayıt haline getirilir. Model sadece cevap sözcüğünü üretir. Serbest Türkçe anlamayı öğrendiği iddia edilmez.

## Kullanım ve kaynaklar

- **Yeni model:** `experiments/dual_upsert.py`
- **Asıl kaydedilmiş checkpoint'lerle doğrulama:** `experiments/validate_dual_upsert.py`
- **Regresyon:** `tests/test_dual_upsert.py`
- **Ham kayıt:** `results/dual_upsert/validation.json`
- **4 gerçek Türkçe güncelleme örneği:** `results/dual_upsert/outputs_tr.md`
- **GitHub Actions:** `.github/workflows/dual_upsert_validation.yml`

```python
import torch
from experiments.dual_upsert import DualAddressUpsertAMN

m = DualAddressUpsertAMN(48)
ckpt = torch.load(
    "results/relational_qa/equal50/dual_0.pt",
    map_location="cpu", weights_only=True
)
m.load_state_dict(ckpt["state_dict"], strict=True)
m.eval()
# predictions = m(structured_facts_tensor, structured_query_tensor)
```

```bash
python -m pytest -q tests/test_dual_upsert.py
python experiments/validate_dual_upsert.py
```

**Proje kararı:** Yeni sürüm ayrı tutuldu. Orijinal Dual ve diğer mimarilerin kodu/checkpoint'leri değiştirilmedi. Upsert doğal dilde `sil` komutu, sınırsız yeni kelimeler ve çok-aşamalı Türkçe sorgu desteği sağlamaz.
