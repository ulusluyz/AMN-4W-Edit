# AMN-4W-Edit — 1.000 adımlık Türkçe metin eğitim hız karşılaştırması

Tarih: **2026-11-11**. Kaynak: yalnızca **AMN-4W-Edit**. Dondurulmuş `AMN-4W-jules` değiştirilmedi.

## Kapsam

Bu çalışma, bağımsız A→B→C proxy yerine **deponun gerçek karakter tabanlı dil modeli** `src/amn4w.py` ile `src/transformer_baseline.py` arasındadır. Karşılaştırma için aynı başlangıç AMN ağırlıklarına sahip ikinci bir model kullandık; yalnızca token-bağımsız `K,V,eta,gamma,Q` projeksiyonlarını döngü öncesine aldık, bellek güncellemesi/okuma sırasını aynen koruduk.

Bu **genel Türkçe dil modelleme başarısı kanıtı değildir**. Deponun `data/train.json` dosyası sadece **6 soru-cevap kaydı** içerir. `data/test_new_knowledge.json` 4 kayıt içerir; **2'si eğitimle aynı metindir**. Karakter tokenizer sözlüğü mevcut eğitim yöntemi gibi eğitim+test metinlerinden oluşturulmuştur (sözlük düzeyinde test bilgisi görüldü). Yeni sorulara genelleme ve gerçek cevap üretimi bu deneyle kanıtlanamaz.

## Kontroller

- 3 seed: `0,1,2`, her biri **1.000 optimizer adımı**, batch **6**, d_model **32**, AMN durum boyutu **32**, karakter dizisi **96**, LR **0,002**, Adam, grad clipping **1,0**.
- Ortam: **PyTorch 2.10.0+cpu**, tek CPU iş parçacığı, CPU. Modelin öğrenilebilir parametreleri: orijinal/optimize AMN **13.278**, Transformer **276.828** (yaklaşık 20,85 kat fazla). Transformer deponun **iki katmanlı, 2 başlıklı** mevcut sınıfı; dropout ayarı orijinalindeki gibi bırakılmıştır.
- Eğitim, orijinal repo `scripts/train.py` içindeki `64` maksimum karakter sınırını **96** olarak değiştirir; çünkü bu sınır 6 örneğin çoğunda cevabı kestiği için kıyaslamayı geçersiz kılardı. Bunun dışında dil modelinin AMN denklemi değişmedi.
- AMN orijinal/optimize aynı seed ile **aynı başlangıç ağırlıkları** ve **aynı batch indeks sırası** kullanır. Transformer da aynı eğitim örneklerini ve aynı batch sırasını kullanır.
- Değerlendirme: **4 QA metninde** öğretmen zorlamalı (golden prefix + cevap karakterleri) çapraz entropi, token doğruluğu, cevap karakteri doğruluğu, cevap dizisi tam eşleşmesi. **Serbest üretim / sohbet başarısı ölçülmemiştir.**
- Farklı model büyüklükleri sebebiyle bu karşılaştırma **eşit parametreli** değildir. Deneme CPU eğitim süresini ve bu küçük göreve uygunluğunu ölçer; genel mimari üstünlüğü değil.

## Ham sonuçlar

| Seed | Model | Eğitim sn | Val loss | Val token doğruluğu | Cevap karakter doğruluğu | Tam cevap |
|---:|---|---:|---:|---:|---:|---:|
| 0 | AMN orijinal | 65,36 | 2,2881 | %70,03 | %52,94 | %50 |
| 0 | AMN optimize | 23,97 | 2,2881 | %70,03 | %52,94 | %50 |
| 0 | Transformer | 46,24 | 0,8982 | %87,54 | %52,94 | %50 |
| 1 | AMN orijinal | 63,18 | 2,0231 | %69,02 | %58,82 | %50 |
| 1 | AMN optimize | 24,53 | 2,0244 | %69,02 | %58,82 | %50 |
| 1 | Transformer | 50,26 | 0,8475 | %88,55 | %64,71 | %50 |
| 2 | AMN orijinal | 66,22 | 2,0692 | %72,05 | %52,94 | %50 |
| 2 | AMN optimize | 26,94 | 2,1064 | %73,06 | %52,94 | %50 |
| 2 | Transformer | 49,17 | 0,9539 | %86,87 | %52,94 | %50 |

### Üç seed ortalaması

| Ölçüm | Orijinal AMN | Optimize AMN | Transformer |
|---|---:|---:|---:|
| Eğitim süresi | 64,92 s | **25,15 s** | 48,55 s |
| Eğitim süresi standart sapması | 1,57 s | 1,58 s | 2,08 s |
| Validation token doğruluğu | %70,37 | %70,71 | **%87,65** |
| Validation cevap karakter doğruluğu | %54,90 | %54,90 | **%56,86** |
| Validation tam cevap eşleşmesi | %50 | %50 | %50 |
| Validation loss | 2,1268 | 2,1397 | **0,8999** |

**Optimize AMN vs orijinal AMN:** **2,58×** hız; ortalama eğitim süresinde **%61,26 azalma**. **Optimize AMN vs Transformer:** **1,93×** hız; sürede **%48,21 azalma**. Ham süreler tekil CPU ölçümleridir; farklı işlemci, iş parçacığı ve model büyüklüklerinde genellenemez.

## Gradyan ve kalite korunması

Aynı seed ve aynı başlangıç parametrelerinde, ileri geçiş ve geri yayılım karşılaştırması:

| Seed | max logit farkı | max S farkı | max gradyan farkı |
|---:|---:|---:|---:|
| 0 | 2,3842e-7 | 9,6858e-8 | 2,9802e-8 |
| 1 | 2,3842e-7 | 1,1921e-7 | 1,4901e-8 |
| 2 | 2,5332e-7 | 1,4901e-7 | 1,4901e-8 |

Bu değerler **sayısal tolerans içinde eşdeğerliği** destekliyor. Eğitim boyunca float32 işlem sırası değişebildiğinden bütün ağırlıklar bit düzeyinde aynı değildir. 3 seed'de AMN cevap karakteri ve tam cevap skoru aynı kaldı; seed 2'de token doğruluğu %72,05'ten %73,06'ya çıktı, val loss da 2,0692'den 2,1064'e yükseldi. Dolayısıyla tüm kalite ölçülerinin tamamen değişmediği **iddia edilmiyor**.

## Sonuç

1. **Performans kazanımı ölçüldü:** mevcut karakter tabanlı AMN dil modeli, yalnızca hesaplama düzeni değiştirilerek bu CPU deneyinde **2,58 kat hızlandı**.
2. **Cevap performansı gerilemedi:** 4 kayıtlık öğretmen zorlamalı değerlendirmede 3 seed'in tamamında cevap karakteri ve tam cevap skorları sabit. Bu değerlendirme **çok küçük**.
3. **Transformer genel token modellemede daha iyi:** daha düşük val loss ve %87,65'e karşı %70,71 token doğruluğu; optimize AMN ise bu donanım/konfigürasyonda 1,93 kat hızlı. Model boyutları çok farklı.
4. **Gerçek uzun metin / genel Türkçe testine ihtiyaç var:** veri çeşitliliği, bağımsız test kümesi, serbest üretim, farklı bağlam uzunlukları ve eşit parametre/hardware budget çalışması henüz yapılmadı.
5. **Varsayılan eğitim kodu ve eski checkpointler değişmedi.** Optimize kod ayrı deney modülünde tutuluyor.

## Tekrar üretim

```bash
python experiments/benchmark_turkish_1000.py --steps 1000 --seeds 0 1 2 --batch 6 --max-len 96 --dim 32 --memory 32 --lr 0.002 --threads 1
python -m pytest -q tests/test_language_speed_parity.py
```

Kaynak kod: [`experiments/benchmark_turkish_1000.py`](../../experiments/benchmark_turkish_1000.py)  
Özet ham kayıtlar: [`results/comparison/turkish_language_1000_cpu_3seed.jsonl`](../../results/comparison/turkish_language_1000_cpu_3seed.jsonl)

Not: Bu çalışmanın yerel yürütmesi, mevcut repo sınıflarının denklemleri transkribe edilerek oluşturulmuş bağımsız, öz-yeterli Python test betiğiyle yapıldı. GitHub'daki kısa tekrar üretim betiği doğrudan `src` modellerini kullanır. GitHub CI'de sayısal regresyon testi geçti; **repo betiğinin 1.000 adımlık tam çıktısının yerel ham kayıtla yeniden eşleştirilmesi henüz yapılmadı**.
