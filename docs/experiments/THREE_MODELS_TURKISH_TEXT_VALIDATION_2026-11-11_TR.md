# Üç model — Türkçe metinle eşit koşullarda doğrulama (11 Kasım 2026)

**Proje:** AMN-4W-Edit. **Durum:** 5 seed ve iki bağımsız test ailesi tamamlandı. Kaynak AMN-4W-jules sadece okundu.

## Test protokolü

Var olan ve kaydedilmiş **50 adımlık checkpoint'ler**, hiçbir ağırlık değiştirilmeden yeniden yüklendi. Standart AMN, Dual AMN ve Transformer aynı kişi–ilişki–değer bağlamlarını ve soruları gördü. Her birinde 5 seed (0–4); seed başına 3.000 görülmemiş ilişkisel soru ve 500 güncelleme sorusu.

Eğitim karşılaştırmasının ortak koşulları: Adam, öğrenme oranı 0,003, batch 128, temsil boyutu 48, dört bilgi içeren bağlam, yalnız cevap çapraz entropisi. Model parametreleri eşit değil: AMN 8.874, Dual AMN 8.874, Transformer 42.666. Bu çalışmada eğitim tekrarlanmadığı için yeni hız kıyaslaması yapılmadı.

**Türkçe metin konusunda önemli sınır:** Gösterilen cümleler Türkçe olarak yazılmış fakat sentetik olgulardır. Girişte hepsine aynı **kural tabanlı kişi–ilişki–değer ayrıştırıcısı** uygulandı. Modellerin kendileri serbest Türkçe anlamıyor ve yalnız tek kelimelik cevap tokeni üretiyor. Bu, doğal dilde uçtan uca anlama testi değil, gerçek ağırlıklardan Türkçe arayüzlü ilişkisel okuma testidir.

## Beş seed sonunda gerçek sonuç

| Görev | Standart AMN | Dual AMN | Transformer |
|---|---:|---:|---:|
| Görülmemiş bağlar | 14.996 / 15.000 | **15.000 / 15.000** | 7.000 / 15.000 |
| Doğruluk | %99,973 | **%100** | %46,667 |
| Tekrarlanan anahtarda son yazılan değeri bulma | **2.500 / 2.500** | 1.256 / 2.500 | 952 / 2.500 |
| Güncelleme doğruluğu | **%100** | %50,24 | %38,08 |

Görülmemiş test, 31 benzersiz test üçlüsü çevresinde örneklenen toplam 15.000 farklı bağlamdır; 15.000 ayrı bilgi demek değildir. Tekrarlanan anahtar stresi, daha önce eğitimde bulunmayan bir görevdir. Aynı kişi–ilişki anahtarının farklı değerleri bağlamın başında ve sonunda verilir; doğru cevap her zaman son kayıttır. Son kayıt sırayla eski veya yeni değer olabilir. Üç model tam olarak aynı 500 örneği/seed gördü.

### Seed bazında

| Seed | AMN temiz | Dual temiz | Transformer temiz | AMN güncelleme | Dual güncelleme | Transformer güncelleme |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2996/3000 | 3000/3000 | 1408/3000 | 500/500 | 250/500 | 221/500 |
| 1 | 3000/3000 | 3000/3000 | 1429/3000 | 500/500 | 249/500 | 178/500 |
| 2 | 3000/3000 | 3000/3000 | 1409/3000 | 500/500 | 262/500 | 222/500 |
| 3 | 3000/3000 | 3000/3000 | 1392/3000 | 500/500 | 256/500 | 168/500 |
| 4 | 3000/3000 | 3000/3000 | 1362/3000 | 500/500 | 239/500 | 163/500 |

## Gerçek checkpoint'lerden Türkçe çıktılar

### Örnek 1 — Doğru ve yanlış tahmin

Can şu anda bahçede. Okan şu anda parkta. Mina'nın kaleminin rengi beyaz. Zeynep şu anda garajda.

**Soru:** Can şu anda nerede?

- Beklenen değer: **bahçe**.
- Standart AMN: **bahçe** (doğru; seçilen sınıf olasılığı 0,9252).
- Dual AMN: **bahçe** (doğru; 0,9789).
- Transformer: **garaj** (yanlış; 0,2958).

### Örnek 2 — Başka bilgiye karışma

Murat'ın kaleminin rengi mavi. Yiğit şu anda salonda. Sena'nın kaleminin rengi sarı. Umut'un kaleminin rengi siyah.

**Soru:** Murat'ın kaleminin rengi nedir?

- Beklenen değer: **mavi**.
- Standart AMN: **mavi**.
- Dual AMN: **mavi**.
- Transformer: **siyah** (yanlış).

### Örnek 3 — Güncellemede Dual AMN hatası

Ayşe'nin kaleminin rengi sarı. Elif şu anda kafede. Sena'nın kaleminin rengi yeşil. **Ayşe'nin kaleminin rengi kırmızı.**

**Soru:** Ayşe'nin kaleminin rengi nedir?

- Beklenen son değer: **kırmızı**.
- Standart AMN: **kırmızı** (doğru).
- Dual AMN: **sarı** (yanlış; eski değer).
- Transformer: **kırmızı** (doğru).

### Örnek 4 — İki farklı hatalı okuma

Umut'un kaleminin rengi siyah. Deniz'in kaleminin rengi beyaz. Mina'nın kaleminin rengi beyaz. **Umut'un kaleminin rengi sarı.**

**Soru:** Umut'un kaleminin rengi nedir?

- Beklenen son değer: **sarı**.
- Standart AMN: **sarı** (doğru).
- Dual AMN: **siyah** (yanlış; eski bilgi).
- Transformer: **beyaz** (yanlış; ilgisiz kişi).

Model çıktısı yalnız tek sözlük kelimesidir. Türkçe soru ve bağlam cümleleri karşılaştırma arabiriminin gösterimidir; modellerin ürettiği cümleler değildir.

## Teknik sorun

Dual bellekte bütün anahtarlar aynı anda Gram sistemiyle çözümleniyor. Aynı kişi–ilişki anahtarı iki kez tutulunca satırlar doğrusal bağımlı hale geliyor. Mevcut yöntem ilk bilgiyi yenisiyle değiştiren bir **upsert / last-write-wins** kuralı içermiyor. Dolayısıyla bağımsız adreslerin iyi korunması, aynı adresin güncellenmesini garanti etmiyor.

Standart AMN, son yazılan değeri izleyebiliyor; fakat bağımsız ama benzer anahtarlar arasında bellek hasarı riski önceki kontrollü testlerde gözlendi. Bu nedenle iki ölçüm ayrı tutulmalı.

## Desteklenmeyen görevler

- Serbest Türkçe paragrafın doğrudan modele verilmesi: **desteklenmiyor**.
- Yeni kişi / yeni kelime (ör. Bora): kapalı sözlük dışında, **desteklenmiyor**.
- Belirli bilginin silinmesi için metinsel komut: **desteklenmiyor**.
- Birden çok ilişkiyi zincirleyip iki aşamada cevaplama: **desteklenmiyor**.
- Serbest metin halinde uzun Türkçe cevap üretimi: **desteklenmiyor**.

Bu görevler için başarı yüzdesi hesaplanmadı; desteklenmeyen formatlar başarısız doğal dil anlama skoruna dönüştürülmedi.

## Yeniden üretim

- Deney: experiments/validate_three_models_turkish.py
- Ham JSON: results/real_turkish_3models/validation.json
- Bütün 12 normal + 4 güncelleme örneği: results/real_turkish_3models/outputs_tr.md
- GitHub Actions: .github/workflows/verify_three_models_turkish.yml
- Checkpointler: results/relational_qa/equal50/

Komut: python experiments/validate_three_models_turkish.py

**Özet:** Korunan bağımsız kayıt okumasında Dual AMN, son-yazma güncellemesinde standart AMN daha iyi. Bu dar görevler genel Türkçe dil becerisi veya Transformer'a karşı genel mimari üstünlük kanıtı değildir.
