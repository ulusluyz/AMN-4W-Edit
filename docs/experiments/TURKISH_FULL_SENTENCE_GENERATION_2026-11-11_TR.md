# AMN-4W-Edit — Gerçek sinir ağıyla Türkçe tam cümle üretimi

**Tarih:** 11 Kasım 2026. **Depo:** AMN-4W-Edit. Eski AMN-4W-jules salt okunur bırakıldı.

## Ne yapıldı?

Üç mevcut 50-adımlık ilişkisel soru-cevap modelinin (`dual_upsert`, `amn`, `transformer`) checkpoint'i değişmeden yüklendi; eğitimden sonra donduruldu. Her modelin üzerine **aynı mimaride**, bağımsız eğitilen GRU tabanlı **oto-regresif kelime çözücüsü** eklendi. Bu yeni çözücü, `<bos>` ile başlayıp kendi önceki tahminlerinden devam ederek dört kelimelik Türkçe cümleyi ve bitiş tokenini çıkarır. **Tahmin sırasında cevap metni şablonla birleştirilmez.**

Fakat eğitimde kullanılan hedef cümleler **sabit ve sınırlı Türkçe kalıplarla** hazırlanmıştır. Ayrıca model girdisi serbest Türkçe paragraf değil, ayrıştırılmış kişi/ilişki/değer üçlüleri ve sorgu alanlarıdır. Bu çalışma henüz genel Türkçe anlayış veya serbest konulu metin üretimi değildir.

## Eşit koşullarda çalışma

Her modelin çözücüsü için seed **0, 1 ve 2**; en fazla 650 adım; Adam 0,003; batch 128; **8.192** eğitim bağlamı; **640** geliştirme bağlamı; **1.200** son test bağlamı kullanıldı. Farklı seed'lerdeki aynı tür veri örnekleme şeması ve eğitim adımları kullanıldı. Geliştirme doğruluğu %100 olduğunda ardışık iki kontrolden sonra durdurma yapıldı. Tüm modeller aynı sayıda örnek ve aynı decoder mimarisiyle eğitildi, ancak **aynı son eğitim adımında durmadılar**. Orijinal model parametreleri eğitilmedi. Ayrıştırılmış veri üçlüleri train=264, dev=25, test=31 şeklinde ayrıdır; testte kelimeler zaten sözlüktedir.

| Model | Tam doğru cümle | Tam cümle doğruluğu | Ek cümle çözücüsü adım/seed |
|---|---:|---:|---|
| **Dual Upsert AMN** | **3.516 / 3.600** | **%97,667** | 400 / 650 / 350 |
| Standart AMN | 3.509 / 3.600 | %97,472 | 400 / 650 / 400 |
| Transformer | 129 / 3.600 | %3,583 | 650 / 650 / 650 |

**Bu yeni yüzde, sınırlı dilbilgili tam cümle eşleşmesidir**; önceki tek kelime ilişki doğruluğunun yerini almaz. Transformer'ın düşük sonucu bu küçük 50-adımlık dondurulmuş soru-cevap checkpoint'ine ve yeni çözücü girişlerine aittir, genel Transformer yeteneğine ait değildir.

## Kaydedilmiş modellerin örnek çıktıları

### Örnek 1

- Bağlamdaki gerçek olgu: Elif konum kafe.
- Sorgu: Elif / konum.
- **Beklenen:** Elif şu anda kafededir.
- **Dual Upsert:** Elif şu anda kafededir.
- **Standart AMN:** Elif şu anda kafededir.
- **Transformer:** Elif şu anda parktadır. (**yanlış**)

### Örnek 2

- Bağlamdaki gerçek olgu: Yiğit renk sarı.
- Sorgu: Yiğit / renk.
- **Beklenen:** Yiğit'in kaleminin rengi sarıdır.
- **Dual Upsert:** Yiğit'in kaleminin rengi sarıdır.
- **Standart AMN:** Yiğit'in kaleminin rengi sarıdır.
- **Transformer:** Yiğit'in kaleminin rengi sarıdır.

### Örnek 3

- Bağlamdaki gerçek olgu: Umut konum garaj.
- Sorgu: Umut / konum.
- **Beklenen:** Umut şu anda garajdadır.
- **Dual Upsert:** Umut şu anda garajdadır.
- **Standart AMN:** Umut şu anda garajdadır.
- **Transformer:** Umut şu anda okuldadır. (**yanlış**)

Bu cümleler yalnız doğruluk metriği değil, yeni modelin **üretim çağrısından** kaydedilen kelime dizileridir. Altın cevap yalnız eğitim ve değerlendirme sırasında kullanıldı; üretim çağrısına verilmedi.

## Arşiv

- Deney kodu: `experiments/turkish_sentence_decoder.py`.
- Regresyonlar: `tests/test_turkish_sentence_decoder.py`.
- 9 yeni decoder ağırlığı: `results/real_turkish_sentences/{dual_upsert,amn,transformer}_{0,1,2}.pt`.
- Seed bazlı metrikler ve model çıktısı: aynı dizindeki `.json` dosyaları.
- Türkçe örnekler: `results/real_turkish_sentences/model_generated_sentences_tr.md`.
- GitHub Actions iş akışı: `.github/workflows/turkish_sentence_generation.yml`.

**Kazanımın sınırı:** Tek kelimelik çıktıdan, gerçek oto-regresif **tam cümle üretimine** geçildi. Genel serbest Türkçe cümle kurma ve kullanıcıdan serbest Türkçe girdi alma **henüz sağlanmadı**. Önceki checkpointler ve model mimarileri değişmedi.
