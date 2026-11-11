# Dual AMN upsert — Türkçe model çıktıları

**Uyarı:** Türkçe cümleler sabit bir biçimlendiriciden gelir.
Modeller yalnız tek kelimelik cevap tahmin eder; serbest Türkçe anlamaz.

## Örnek 1
Ayşe'nin kaleminin rengi sarı. Elif şu anda kafede. Sena'nın kaleminin rengi yeşil. Ayşe'nin kaleminin rengi kırmızı.
**Soru:** Ayşe'nin kaleminin rengi nedir?
**Beklenen son değer:** kırmızı
- Standart AMN: **kırmızı**
- Eski Dual AMN: **sarı**
- Upsert Dual AMN: **kırmızı**

## Örnek 2
Umut'un kaleminin rengi siyah. Deniz'in kaleminin rengi beyaz. Mina'nın kaleminin rengi beyaz. Umut'un kaleminin rengi sarı.
**Soru:** Umut'un kaleminin rengi nedir?
**Beklenen son değer:** sarı
- Standart AMN: **sarı**
- Eski Dual AMN: **siyah**
- Upsert Dual AMN: **sarı**

## Örnek 3
Okan şu anda bahçede. Zeynep şu anda kafede. Can şu anda bahçede. Okan şu anda parkta.
**Soru:** Okan şu anda nerede?
**Beklenen son değer:** park
- Standart AMN: **park**
- Eski Dual AMN: **bahçe**
- Upsert Dual AMN: **park**

## Örnek 4
Okan şu anda parkta. Yaren şu anda okulda. Ayşe'nin kaleminin rengi sarı. Okan şu anda okulda.
**Soru:** Okan şu anda nerede?
**Beklenen son değer:** okul
- Standart AMN: **okul**
- Eski Dual AMN: **park**
- Upsert Dual AMN: **okul**

## Kümülatif başarı

| Model | Güncelleme | 4 kayıt | 6 kayıt | 8 kayıt | 12 kayıt |
|---|---:|---:|---:|---:|---:|
| amn_original | 2500/2500 | 14996/15000 | 4994/5000 | 4990/5000 | 4919/5000 |
| dual_original | 1256/2500 | 15000/15000 | 5000/5000 | 5000/5000 | 5000/5000 |
| dual_upsert | 2500/2500 | 15000/15000 | 5000/5000 | 5000/5000 | 5000/5000 |

**Aynı ağırlıklar:** Dual upsert yeni parametre eklemedi; sadece tekrarlanan anahtarların eski kayıtlarını Gram çözümünden çıkardı.
Tekrarlanan anahtar bulunmadığında modelin logitleri bit düzeyinde aynıdır.
