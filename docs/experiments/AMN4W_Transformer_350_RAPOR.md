# AMN-4W ile Transformer: kontrollü sentetik karşılaştırma

**2026-11-11 — PyTorch 2.10.0+cpu; deneysel AMN proxy; üretim AMN-4W v0.4 değil.**

## Kurulum

- Görev: 2 bağımsız A→B→C zincirinden verilen A için C cevabını bulma.
- Eğitim: her mimari için aynı seed ve örnekler; 350 Adam adımı, lr=0,002, batch=64, V=50, M=2, 64 boyut.
- Değerlendirme: her seed için aynı 500 görülmemiş rastgele örnek; 5 seed.
- Transformer: 2 encoder katmanı, 4 dikkat başlığı, FFN=128, dropout=0.
- AMN: delta matris bellek, açık write-mask, iki-aşamalı okuma. AMN_aux ayrı yardımcı ara-adım kaybı kullanır; Transformer yalnız cevap kaybı kullanır.
- Eğitim zamanı tek CPU işçisinde ölçüldü; mimari parametre sayıları eşit değil.

## Gerçek sonuçlar

| Model | Parametre | Nihai doğruluk (%; ort±ss) | İlgisiz zincir kaldırılınca doğru (%) | Hedef zincir kaldırılınca doğru (%) | İlgisiz değişim: doğru cevabın olasılığındaki ort. mutlak fark | Eğitim zamanı (sn; ort±ss) |
|---|---:|---:|---:|---:|---:|---:|
| amn | 31988 | 99.40 ± 1.23 | 100.00 | 1.20 | 0.0438 | 4.03 ± 0.30 |
| amn_aux | 31988 | 100.00 ± 0.00 | 100.00 | 0.56 | 0.0135 | 3.91 ± 0.17 |
| transformer | 73842 | 75.96 ± 22.12 | 100.00 | 0.08 | 0.2699 | 2.33 ± 0.05 |

### Seed bazında doğruluk

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| amn | 100.0% | 99.8% | 100.0% | 97.2% | 100.0% |
| amn_aux | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| transformer | 55.2% | 50.8% | 80.0% | 96.4% | 97.4% |

## Yorum ve sınırlamalar

- Bu **tek bir sentetik görevde, 350 adım bütçesinde** AMN proxy lehine performans kanıtıdır. Dili anlama/üretme başarısı veya genel mimari üstünlük iddiası değildir.
- Transformer 2,3 kat fazla parametreye sahiptir; dolayısıyla parametre-başına veya FLOP-başına dengeli bir kıyas değildir. Aynı adım/batch/örnek garantisi, eşit parametre, FLOP veya gerçek süre garantisi değildir.
- AMN_aux, Transformer’da bulunmayan ara ilişki hedefleriyle öğretildi; en doğrudan kontrol AMN (yardımcı kayıpsız) ile Transformer arasındadır.
- AMN modeline gerçek ilişki sınırlarını sağlayan write-mask verildi; Transformer’a maske verilmedi. Bu, AMN lehine görev yapısı bilgisidir.
- İlgisiz zincirin sıfıra çevrilmesi dağılım dışı bir müdahaledir; sonuç **matris anahtar sızıntısı oranı** ile karşılaştırılamaz. Transformer’ın olasılık değişimi hatalı cevap oranı değildir.
- Hedef zincir kaldırıldığında her iki modelin doğruluğu şansa yakın düşer; bu müdahale bilgi bağımlılığını destekler fakat genelleme kanıtlamaz.
- 1000-adımlık ikinci deney zaman sınırı nedeniyle tamamlanmadı. İki AMN seed sonucu mevcuttur; Transformer için tamamlanmış 1000-adım sonucu olmadığından 1000-adımda mimari karşılaştırma yapılmamaktadır.
- Tam gerçek AMN-4W v0.4, benzer parametreli Transformer ve gerçek dil verisi ile kıyaslaması henüz yoktur.

## Dosyalar

- `compare_amn_transformer.py`: eğitim/değerlendirme ve müdahale testlerinin çalıştırılabilir kodu.
- `amn_transformer_results_350.jsonl`: 15/15 koşunun ham metrikleri.
- `amn_transformer_results_1000.jsonl`: tamamlanmamış 1000-adım denemesi, karşılaştırma için kullanılmaz.