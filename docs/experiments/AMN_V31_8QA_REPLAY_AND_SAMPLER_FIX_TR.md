# AMN v3.1 — Eğitim kapsamı ve unutma düzeltmesi, ölçümlü sonuçlar

**Tarih:** 12 Kasım 2026

## 1. Başlıca hata: Yetersiz soru-cevap gösterimi

Önceki 1.420 adımlık, 16.380.518 parametreli Türkçe AMN eğitiminde:
- Kodlanabilir train havuzu: 17.170 soru-cevap.
- Cevap SFT adımı: 900; batch: 4, toplam 3.600 çekiliş.
- RNG/provenance geri oynatmasında 3.265 farklı soru çıktı.
- Dolayısıyla 13.905 soru, cevap-loss'uyla **hiç eğitilmedi**.
- Eski örneklerin bir kısmı sadece 2–3 kere görüldü.

Ek veri sorunları: bozulan Türkçe kaynak cümleleri ve yanlış matematik referansları.
Yardımcı intent/contrastive başlıklarının skoru decoder çıktısına doğrudan
bağlı değil; yardımcı loss'ları cevap üretiminin doğruluğunu garanti etmiyor.

## 2. Kaynak veriyle doğrulanmış 6-soruluk düzeltme

Bağımsız orijinal veri manifesti SHA, tokenizer SHA ve stochastic SGD
örneklerinin yeniden oynatılmasıyla kaynak doğrulandı.

360 yeni optimizer adımı × batch 3 ile altı kaynak sorusu eşit tekrarlandı:
Her biri 180 kez cevap loss'u aldı.

| Ölçüm | Önce | Sonra |
|---|---:|---:|
| Kaynak cevapla birebir tam üretim | 0/6 | **6/6** |
| Ortalama teacher-forcing NLL | ~6,78 | **0,0030** |
| Doğru soruyu yanlış soruya göre tercih etme | 4/6 | **6/6** |
| Ortalama yanlış-prompt eksi doğru-prompt NLL farkı | 0,000206 | **0,468471** |

Bu, altı soru için modelin hem token üretmeyi hem de soru–cevap bağıntısını
öğrenebildiğini gösterdi. **Serbest Türkçe genelleme kanıtı değildir.**

## 3. Unutmanın ölçümü ve deneysel çözüm

İlk altı sorunun dışında kalan iki kontrol sorusu, altı-soruluk düzeltmeden
sonra geriledi:
- Kontrol teacher-forced NLL ortalaması yaklaşık 6,74 → 8,93.
- Model kontrol sorularına yeni ezberlediği alakasız cevapları üretmeye başladı.

Bu nedenle **experience replay** uygulandı: Altı eski soru ve iki kontrol
sorusu aynı 8-soruluk havuzda dengeli tekrar eğitildi.

Yeni deneme: 320 optimizer adımı × batch 4. Sekiz sorunun her biri 160 kere
gerçek gradient-loss hesabına dahil edildi.

| Ölçüm | 8'li replay öncesi | 8'li replay sonrası |
|---|---:|---:|
| Önceki altı cevap tam üretimi | 6/6 | **6/6** |
| Eklenen iki cevap tam üretimi | 0/2 | **2/2** |
| Eski altı cevap NLL | 0,003014 | **0,001403** |
| Eklenen iki cevap NLL | 8,926338 | **0,002395** |

Ağırlık ve tokenizer kaynakları değişmedi. Yeni checkpoint,
results/amn_tm_v31_replay8/trained.pt yolunda ayrıdır.

**Bu küçük veri üzerinde yerel düzeltmedir.** Diğer 17.000'den fazla
soru için kalite veya öğrenme başarısı iddia edilmez.

## 4. Gelecek eğitimler için kod düzeltmesi

experiments/train_amn_tm_v3.py dosyasına:
- BalancedEpochSampler: her örnek görülmeden aynı örnek tekrarlanmayacak
  şekilde epok ilerlemesi.
- --sft-sampler epoch: dengeli SFT örneklemesini etkinleştirme.
- --require-sft-exposures N: gerçek eğitim sonunda bütün sorular en az N
  kez görülmemişse bilimsel kalite kapısından başarısız geçme.
- sft_coverage metrikleri: toplam seçim, farklı soru sayısı, kapsam
  oranı, asgari ve azami gösterim.

Eski v3.1 koşusunun yeniden üretimi için default random korundu;
düzeltilmiş sonraki koşularda epoch açıkça seçilmeli.

Yeni örnekleyici birim testleri: GitHub Actions 43178149965 — SUCCESS.

## Kanıt ve dosyalar

- 6 QA: results/amn_tm_v31_rehearsal/metrics.json
- 6 QA gerçek cevapları: results/amn_tm_v31_rehearsal/raw_seen_qa_tr.md
- Doğru/yanlış soru swap testi: results/amn_tm_v31_rehearsal/question_conditioning.json
- 8 QA replay: results/amn_tm_v31_replay8/metrics.json
- 8 QA gerçek cevapları: results/amn_tm_v31_replay8/replay_results_tr.md
- Eğitim kapsamı kodu: experiments/train_amn_tm_v3.py
- 8 QA koşusu: GitHub Actions 43177927261 — SUCCESS.

**Sınır:** Model bu sekiz soru-cevabı ezberleyebiliyor ve doğru soruya göre
eşleştirebiliyor; henüz insan düzeyinde veya genel Türkçe anlayışına sahip değil.
