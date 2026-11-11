# Üç eğitilmiş modelin Türkçe çıktıları — gerçek ağırlıklardan

**UYARI:** Türkçe metinler ortak kural tabanlı ayrıştırıcıyla kişi–ilişki–değer üçlülerine dönüştürülmüştür. Modeller serbest Türkçe okumamıştır. Altın cevaplar yalnızca değerlendirme içindir.
Modellerin ürettiği **yalnızca tek kelimelik cevap tokenlarıdır**; bağlam ve soru cümleleri dış arabirim tarafından biçimlendirilmiştir.

## Örnek 1
Murat'ın kaleminin rengi mavi. Ece şu anda okulda. Yaren şu anda parkta. Nisa'nın kaleminin rengi sarı.

**Soru:** Nisa'nın kaleminin rengi nedir?
**Gerçek cevap:** sarı
- amn: **sarı** (DOĞRU; olasılık 0.9759)
- dual: **sarı** (DOĞRU; olasılık 0.9753)
- transformer: **sarı** (DOĞRU; olasılık 0.5834)

## Örnek 2
Can şu anda bahçede. Okan şu anda parkta. Mina'nın kaleminin rengi beyaz. Zeynep şu anda garajda.

**Soru:** Can şu anda nerede?
**Gerçek cevap:** bahçe
- amn: **bahçe** (DOĞRU; olasılık 0.9252)
- dual: **bahçe** (DOĞRU; olasılık 0.9789)
- transformer: **garaj** (YANLIŞ; olasılık 0.2958)

## Örnek 3
Deniz şu anda okulda. Elif şu anda kafede. Sena'nın kaleminin rengi sarı. Deniz'in kaleminin rengi beyaz.

**Soru:** Deniz'in kaleminin rengi nedir?
**Gerçek cevap:** beyaz
- amn: **beyaz** (DOĞRU; olasılık 0.9789)
- dual: **beyaz** (DOĞRU; olasılık 0.9780)
- transformer: **beyaz** (DOĞRU; olasılık 0.4217)

## Örnek 4
Umut'un kaleminin rengi siyah. Okan şu anda parkta. Umut şu anda garajda. Can şu anda salonda.

**Soru:** Umut şu anda nerede?
**Gerçek cevap:** garaj
- amn: **garaj** (DOĞRU; olasılık 0.9622)
- dual: **garaj** (DOĞRU; olasılık 0.9655)
- transformer: **garaj** (DOĞRU; olasılık 0.2871)

## Örnek 5
Murat'ın kaleminin rengi mavi. Yiğit şu anda salonda. Sena'nın kaleminin rengi sarı. Umut'un kaleminin rengi siyah.

**Soru:** Murat'ın kaleminin rengi nedir?
**Gerçek cevap:** mavi
- amn: **mavi** (DOĞRU; olasılık 0.9501)
- dual: **mavi** (DOĞRU; olasılık 0.9674)
- transformer: **siyah** (YANLIŞ; olasılık 0.6801)

## Örnek 6
Yiğit şu anda salonda. Yiğit'in kaleminin rengi turuncu. Nisa'nın kaleminin rengi sarı. Umut'un kaleminin rengi siyah.

**Soru:** Yiğit'in kaleminin rengi nedir?
**Gerçek cevap:** turuncu
- amn: **turuncu** (DOĞRU; olasılık 0.9593)
- dual: **turuncu** (DOĞRU; olasılık 0.9671)
- transformer: **siyah** (YANLIŞ; olasılık 0.3750)

## Örnek 7
Mert şu anda kafede. Umut şu anda garajda. Umut'un kaleminin rengi siyah. Ece şu anda okulda.

**Soru:** Umut şu anda nerede?
**Gerçek cevap:** garaj
- amn: **garaj** (DOĞRU; olasılık 0.9658)
- dual: **garaj** (DOĞRU; olasılık 0.9655)
- transformer: **kafe** (YANLIŞ; olasılık 0.4551)

## Örnek 8
Okan şu anda parkta. Yiğit'in kaleminin rengi turuncu. Yaren şu anda parkta. Zeynep şu anda garajda.

**Soru:** Okan şu anda nerede?
**Gerçek cevap:** park
- amn: **park** (DOĞRU; olasılık 0.9555)
- dual: **park** (DOĞRU; olasılık 0.9649)
- transformer: **park** (DOĞRU; olasılık 0.5218)

## Örnek 9
Sena şu anda salonda. Deniz şu anda okulda. Sena'nın kaleminin rengi sarı. Can şu anda salonda.

**Soru:** Sena şu anda nerede?
**Gerçek cevap:** salon
- amn: **salon** (DOĞRU; olasılık 0.9255)
- dual: **salon** (DOĞRU; olasılık 0.9754)
- transformer: **salon** (DOĞRU; olasılık 0.6032)

## Örnek 10
Zeynep şu anda garajda. Selin şu anda parkta. Deniz'in kaleminin rengi beyaz. Deniz şu anda okulda.

**Soru:** Deniz şu anda nerede?
**Gerçek cevap:** okul
- amn: **okul** (DOĞRU; olasılık 0.9643)
- dual: **okul** (DOĞRU; olasılık 0.9623)
- transformer: **garaj** (YANLIŞ; olasılık 0.4007)

## Örnek 11
Murat'ın kaleminin rengi sarı. Sena'nın kaleminin rengi sarı. Yiğit şu anda salonda. Sena şu anda salonda.

**Soru:** Sena'nın kaleminin rengi nedir?
**Gerçek cevap:** sarı
- amn: **sarı** (DOĞRU; olasılık 0.9760)
- dual: **sarı** (DOĞRU; olasılık 0.9753)
- transformer: **sarı** (DOĞRU; olasılık 0.7174)

## Örnek 12
Murat'ın kaleminin rengi sarı. Nisa'nın kaleminin rengi sarı. Sena şu anda salonda. Okan şu anda parkta.

**Soru:** Nisa'nın kaleminin rengi nedir?
**Gerçek cevap:** sarı
- amn: **sarı** (DOĞRU; olasılık 0.9738)
- dual: **sarı** (DOĞRU; olasılık 0.9753)
- transformer: **sarı** (DOĞRU; olasılık 0.7406)

## Aynı bilgiyi güncelleme: Türkçe örnekler
Bu testte aynı kişi ve ilişki iki kez yazılıyor; hedef son kayıttır.

### Güncelleme örneği 1
Ayşe'nin kaleminin rengi sarı. Elif şu anda kafede. Sena'nın kaleminin rengi yeşil. Ayşe'nin kaleminin rengi kırmızı.
**Soru:** Ayşe'nin kaleminin rengi nedir?
**Son kayıt (beklenen cevap):** kırmızı
- amn: **kırmızı** (DOĞRU; olasılık 0.9715)
- dual: **sarı** (YANLIŞ; olasılık 0.4724)
- transformer: **kırmızı** (DOĞRU; olasılık 0.5579)

### Güncelleme örneği 2
Umut'un kaleminin rengi siyah. Deniz'in kaleminin rengi beyaz. Mina'nın kaleminin rengi beyaz. Umut'un kaleminin rengi sarı.
**Soru:** Umut'un kaleminin rengi nedir?
**Son kayıt (beklenen cevap):** sarı
- amn: **sarı** (DOĞRU; olasılık 0.9759)
- dual: **siyah** (YANLIŞ; olasılık 0.5731)
- transformer: **beyaz** (YANLIŞ; olasılık 0.2778)

### Güncelleme örneği 3
Okan şu anda bahçede. Zeynep şu anda kafede. Can şu anda bahçede. Okan şu anda parkta.
**Soru:** Okan şu anda nerede?
**Son kayıt (beklenen cevap):** park
- amn: **park** (DOĞRU; olasılık 0.9661)
- dual: **bahçe** (YANLIŞ; olasılık 0.5832)
- transformer: **park** (DOĞRU; olasılık 0.3867)

### Güncelleme örneği 4
Okan şu anda parkta. Yaren şu anda okulda. Ayşe'nin kaleminin rengi sarı. Okan şu anda okulda.
**Soru:** Okan şu anda nerede?
**Son kayıt (beklenen cevap):** okul
- amn: **okul** (DOĞRU; olasılık 0.9643)
- dual: **park** (YANLIŞ; olasılık 0.4878)
- transformer: **park** (YANLIŞ; olasılık 0.4319)

## Sayısal sonuçlar
- amn: 4 kayıtlı held-out **14996/15000**, tekrarlanan anahtar güncelleme stresi **2500/2500**.
- dual: 4 kayıtlı held-out **15000/15000**, tekrarlanan anahtar güncelleme stresi **1256/2500**.
- transformer: 4 kayıtlı held-out **7000/15000**, tekrarlanan anahtar güncelleme stresi **952/2500**.

**Serbest Türkçe / yeni sözcük / silme / iki-hop:** Bu checkpointlerde desteklenmiyor. Desteklenmeyen görevde başarı oranı uydurulmamıştır.
