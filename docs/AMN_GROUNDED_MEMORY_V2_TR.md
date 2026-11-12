# AMN — Mevcut Türkçe veriyle soru–bilgi eşleştirme (v3.2)

## Değişiklikler

Bu sürüm mevcut 16.380.518 parametreli GRU+AMN modelini korur.
Veri yeniden indirilmedi. Kaynak:
`data/amn_world_v1/train.jsonl` (746 QA, 243 konu).
Diğer iki küme (`val.jsonl`, `test.jsonl`) yanıt belleğine KESİNLİKLE eklenmez.

### Soru yanıtı

1. Sorunun Türkçe karakterlerini ve yazım noktalamasını normalize eder.
2. Sorunun niyetini yalnız mevcut tanım/kimlik kalıpları kapsamında belirler.
3. Aynı eğitim sorusu veya TAM konu başlığı eşleşiyorsa, ait olduğu bilgiyi bulur.
4. Var olan AMN `query_head` ve `evidence_head` başlıkları gerçek, farklı-konulu
   pozitif/negatif eşleştirme çiftleriyle eğitilir; gerektiğinde aynı konu
   adayları arasında sinirsel sıralama yapar.
5. Yanıt kaynak metninden VERBATIM döndürülür. **Bu yanıt, modelin serbest
   üretimi değildir.** Böylece alakasız ülke tanımları, sözcük döngüleri ve
   tutarsız açıklamalar cevap olarak sunulmaz.
6. Eşleşme yoksa `Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.`
   yanıtını verir. Bilinmeyeni tahmin etmeye çalışmaz.

### Öğrenme

Yeni model, **eski dünya bilgisi checkpoint'inden devam eder**;
6 tam epok hedeflenir, 2 üst üste doğrulama iyileşmesi olmayan epokta
erken durdurma uygulanır. 100 epok bir varsayım değildir. Örnekler
dengeli seçilir; mevcut temel kavramlar deneysel olarak ek tekrar görür.
Bağımsız konu-tutmalı doğrulama/test metrikleri kaydedilir.

### Kaynaktan örnekler

**Kedi nedir?**
Kedi (Felis catus), ayrıca evcil kedi veya ev kedisi olarak da adlandırılır,
küçük, evcilleştirilmiş etobur memelidir.

**Köpek nedir?**
Köpek (Canis lupus familiaris); köpekgiller (Canidae) familyasına ait,
görünüş ve büyüklükleri farklı 400'den fazla ırkı olan, etçil bir memelidir.

**Denklem nedir?**
Denklem, iki niceliğin eşitliğini gösteren bağıntıdır.

**İnsan nedir?**
İnsan (Homo sapiens lit. "bilge insan") veya modern insan, primatların en
yaygın türüdür.

**20 tane erkek insan 8 tane kadın insan ismi yaz**
Erkek isimleri: Alparslan, Göktuğ, Metehan, Yusuf, Kerem, Miran,
Ömer Asaf, Ömer, Atlas, Aras, Miraç, Aslan, Kuzey, Poyraz,
Alperen, Yiğit, Eymen, Aybars, Ali Asaf, Çınar.
Kız isimleri: Alya, Defne, Gökçe, Zeynep, Asel, Umay, Asya, Elisa.

Kaynak bağlantıları ve ret raporu:
`data/amn_world_v1/DATA_QUALITY_TR.md`;
her kaydın `source_url` alanı.

### Yerel kullanım

```bash
python experiments/ask_amn_grounded_v2.py "Kedi nedir?"
python experiments/ask_amn_grounded_v2.py "Salata nasıl yapılır"
python experiments/ask_amn_grounded_v2.py --inspect-raw "Köpek nedir?"
```

`--inspect-raw` serbest sinir ağı cevabını yalnız tanı amaçlı gösterir;
kaynaklı bellek yanıtından ayrı etiketler.

## Bilinen sınır

Bu mimari **öğretilmiş temel bilgiye kontrollü erişimi düzeltir**. Kaynak
içeriğinin ötesinde akıl yürütme, yeni konuyu tanıma, procedür üretme veya
serbest diyalog becerisini kanıtlamaz. Bellekte olmayan bilgiyi 100 tur
tekrarla eğitmek mümkün değildir.

**Modelin orijinal checkpoint'leri korunur.**
