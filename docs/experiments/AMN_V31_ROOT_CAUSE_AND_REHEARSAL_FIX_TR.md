# AMN-TM v3.1 — Kök neden ve kontrollü düzeltme (12 Kasım 2026)

## Sorun

AMN-TM v3.1, kendisinin eğitim sürecinde gördüğü Türkçe sorulara bile tekrarlı ve ilgisiz yanıt veriyordu.

### Kod ve kayıt düzeyinde doğrulanan başlıca neden

- Asıl eğitim havuzu: 17.170 kodlanmış soru-cevap.
- 1.420 optimizasyon adımının yalnızca 900'ünde SFT cevap kaybı uygulandı.
- Batch size 4 nedeniyle cevap eğitimi için toplam 3.600 örnek seçimi yapıldı.
- Kaynak ve RNG yeniden oynatılmasıyla, bu 3.600 seçimin yalnızca 3.265 farklı soruya ait olduğu doğrulandı.
- Dolayısıyla **en az 13.905 soru cevap eğitiminde hiç seçilmedi**. Eğitimde gördüğü birkaç örnek de çoğunlukla yalnız 1–3 kez kullanıldı.
- 16,38 milyon parametre / 8.192 alt sözcüklü model, 258.188 toplam LM hedef tokenı ile eğitildi. Bu sayı çok sınırlıdır.
- Eğitim verisinde hatalı cevaplar mevcut (örneğin x^3'ün 0–5 integrali yanlış etiketlenmiş); tüm kaynağın doğruluğu teyit edilmiş değildir.
- Yardımcı niyet etiketleri sözcük eşlemeli heuristik kurallarla üretildi, insan tarafından doğrulanmış niyet değildir.
- Yardımcı retrieval/intent başlıkları modelin cevap üreten step() yolunda doğrudan kullanılmıyor. Bu nedenle "grounding" kaybı tek başına cevapların doğru bilgiyle üretileceğini garanti etmez.

## Kontrollü deneysel düzeltme

Mevcut v3.1 checkpoint'i **değiştirilmeden** ayrı bir kopya ile:

- Gerçek Türkçe veri kümesinden, daha önce SFT sırasında gerçekten seçildiği SHA ve RNG replay ile doğrulanmış, kaynak cevapları açık 6 soru seçildi.
- İki başka gerçek soru yeni tekrar eğitimine alınmadı; unutma kontrolü oldu.
- 360 yeni optimizasyon adımı, batch=3, eşit dağılımlı epoch/tekrar seçimi.
- Altı sorunun her biri 180 defa cevap loss'unda kullanıldı.
- Model yanıtları greedy/otoregresif üretildi; cevap lookup veya elle düzeltme yok.

## Gerçek ölçüm

| Ölçüm | v3.1 eski ağırlık | Kontrollü tekrar sonrası |
|---|---:|---:|
| 6 eğitim sorusunda birebir tam yanıt | 0/6 | **6/6** |
| 6 eğitim sorusunda ortalama teacher-forced NLL | yaklaşık 6,78 | **0,0030** |
| Yeni eğitime girmeyen 2 kontrol sorusunda ortalama NLL | yaklaşık 6,74 | **8,93** |
| Yeni eğitime girmeyen 2 kontrol sorusunda tam yanıt | 0/2 | **0/2** |

Başarılı örnek: “Doğru bir çizgi için matematiksel denklemi türetin.” → “y = mx + b, burada m eğim ve b çizginin y-kesiti olarak tanımlanır.”

Bu deney **hesaplamanın eğitime cevap verdiğini ve modelin altı örneği öğrenebildiğini** kanıtlar. **Genel Türkçe anlamayı, bilgi genellemesini veya daha önce görülmemiş soruları çözmeyi kanıtlamaz.**

## Yeni tespit: Dar kümeye aşırı uyum ve unutma

İki kontrol sorusundaki NLL'nin kötüleşmesi, sadece altı örnekte çok güçlü öğrenme yapmanın genel beceriyi iyileştirmediğini gösteriyor. Kontrol sorularında model bazı öğretilmiş cevapları alakasız biçimde tekrar üretti.

Bu nedenle sadece aynı veriyi yüzlerce kez tekrar etmek tüm sorunu çözmez. Düzeltme yöntemi:

1. Gerçek cevap verisini doğruluk/saflık kontrollerinden geçir.
2. Eğitim kapsamını ölç, her örnek için asgari gösterimi raporla, epok tabanlı dengeli örnekleme uygula.
3. SFT eğitimi sırasında farklı konulardaki eski veri örneklerini birlikte tekrar et (experience replay).
4. Aşamalı genişleyen kümelerde hem öğrenilen soruların gerçek token üretimini hem kontrol sorularındaki kaybı izle.
5. Bağımsız ve insan etiketli test ile anlam/niyet başarısı ölçülmeden genel başarı iddiasında bulunma.

## Kaynaklar

- experiments/fix_amn_v31_rehearsal.py
- tests/test_amn_v31_rehearsal_fix.py
- results/amn_tm_v31_rehearsal/metrics.json
- results/amn_tm_v31_rehearsal/raw_seen_qa_tr.md
- results/amn_tm_v31/seen_training_questions/audit.json
- GitHub Actions run 43176442536 (success)

Eski tüm checkpointler korunmuştur.
