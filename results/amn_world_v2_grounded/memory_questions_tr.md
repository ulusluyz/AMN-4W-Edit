# AMN v3.1 — Bilgi belleği, soru eşleştirme ve cevap koruması

Kaynak yalnızca mevcut temiz eğitim verisidir; veri indirimi yapılmadı.
Doğrulanmış bir kayıt bulunursa cevap doğrudan BELLEKTEKİ KAYNAK metnidir.
Bu cevaplar sinir ağının serbest metin üretimi olarak sunulmamıştır.
Eşleşme yoksa alakasız kelimeler üretmek yerine bilgi yok denir.

Eğitim turu: 2/6
Yeni SFT adımı: 374
SFT doğrulama kaybı: 6.7478 -> 6.7478
Bağımsız test kaybı: 6.8042 -> 6.8042
Öğrenilmiş kanıt eşleştirme val in-batch recall@1: 0.19047619047619047 -> 0.47619047619047616

## Veri kümesinde gerçekten ne var?

**Köpek hakkında kısa bilgi verir misin?**
Köpek (Canis lupus familiaris); köpekgiller (Canidae) familyasına ait, görünüş ve büyüklükleri farklı 400'den fazla ırkı olan, etçil bir memelidir.
https://tr.wikipedia.org/wiki/K%C3%B6pek

**Kedi hakkında kısa bilgi verir misin?**
Kedi (Felis catus), ayrıca evcil kedi veya ev kedisi olarak da adlandırılır, küçük, evcilleştirilmiş etobur memelidir.
https://tr.wikipedia.org/wiki/Kedi

**İnsan hakkında kısa bilgi verir misin?**
İnsan (Homo sapiens lit. "bilge insan") veya modern insan, primatların en yaygın türüdür.
https://tr.wikipedia.org/wiki/%C4%B0nsan

**Denklem hakkında kısa bilgi verir misin?**
Denklem, iki niceliğin eşitliğini gösteren bağıntıdır.
https://tr.wikipedia.org/wiki/Denklem

**Erkek hakkında kısa bilgi verir misin?**
Erkek, yetişkin bir eril insandır.
https://tr.wikipedia.org/wiki/Erkek

**Kadın hakkında kısa bilgi verir misin?**
Kadın, yetişkin bir dişi insandır.
https://tr.wikipedia.org/wiki/Kad%C4%B1n


## Kullanıcı soruları ve belleğin yanıtları

**Kedi nedir?**
Kedi (Felis catus), ayrıca evcil kedi veya ev kedisi olarak da adlandırılır, küçük, evcilleştirilmiş etobur memelidir.
Eşleştirme: exact_train_question; konu: Kedi; kaynak: https://tr.wikipedia.org/wiki/Kedi

**Köpek nedir?**
Köpek (Canis lupus familiaris); köpekgiller (Canidae) familyasına ait, görünüş ve büyüklükleri farklı 400'den fazla ırkı olan, etçil bir memelidir.
Eşleştirme: exact_train_question; konu: Köpek; kaynak: https://tr.wikipedia.org/wiki/K%C3%B6pek

**Denklem nedir?**
Denklem, iki niceliğin eşitliğini gösteren bağıntıdır.
Eşleştirme: exact_train_question; konu: Denklem; kaynak: https://tr.wikipedia.org/wiki/Denklem

**İnsan nedir?**
İnsan (Homo sapiens lit. "bilge insan") veya modern insan, primatların en yaygın türüdür.
Eşleştirme: exact_train_question; konu: İnsan; kaynak: https://tr.wikipedia.org/wiki/%C4%B0nsan

**Erkek nedir?**
Erkek, yetişkin bir eril insandır.
Eşleştirme: exact_train_question; konu: Erkek; kaynak: https://tr.wikipedia.org/wiki/Erkek

**Kadın nedir?**
Kadın, yetişkin bir dişi insandır.
Eşleştirme: exact_train_question; konu: Kadın; kaynak: https://tr.wikipedia.org/wiki/Kad%C4%B1n

**20 tane erkek insan 8 tane kadın insan ismi yaz**
Erkek isimleri: Alparslan, Göktuğ, Metehan, Yusuf, Kerem, Miran, Ömer Asaf, Ömer, Atlas, Aras, Miraç, Aslan, Kuzey, Poyraz, Alperen, Yiğit, Eymen, Aybars, Ali Asaf, Çınar. Kız isimleri: Alya, Defne, Gökçe, Zeynep, Asel, Umay, Asya, Elisa.
Eşleştirme: exact_train_question; konu: 2025 resmi isimler iki grup; kaynak: https://nip.tuik.gov.tr/?value=YeniDoganIsimleri

**Cumhuriyet nedir**
Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.
Eşleştirme: unknown_topic; konu: bilinmiyor; kaynak: yok

**Güneş nedir**
Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.
Eşleştirme: unknown_topic; konu: bilinmiyor; kaynak: yok

**Salata nasıl yapılır**
Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.
Eşleştirme: unsupported_intent; konu: bilinmiyor; kaynak: yok

**Saksı nedir**
Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.
Eşleştirme: unknown_topic; konu: bilinmiyor; kaynak: yok

**Pusula nedir**
Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.
Eşleştirme: unknown_topic; konu: bilinmiyor; kaynak: yok

**Kedi nasıl kullanılır**
Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor.
Eşleştirme: unsupported_intent; konu: bilinmiyor; kaynak: yok
