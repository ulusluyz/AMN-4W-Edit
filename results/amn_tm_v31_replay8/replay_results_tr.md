# AMN v3.1 — sekiz gerçek soruyla unutmayı azaltma deneyi

Önceki altı öğrenilmiş soru ve eski iki kontrol sorusu birlikte
dengeli tekrar eğitimine alındı. Orijinal ve altı-soruluk checkpointler
değiştirilmedi. Çıktılar modelin gerçek otoregresif üretimidir.

Bu **görülmüş örnek / hafızada tutma** denetimidir; yeni sorulara
genelleme veya genel Türkçe yeteneği kanıtlamaz.

Eski altı sorunun önce tam cevabı: 6/6
Eski altı sorunun sonra tam cevabı: 6/6
Önceki iki kontrolün önce tam cevabı: 0/2
Önceki iki kontrolün sonra tam cevabı: 2/2

## Soru 1
**Girdi:** Bir konuşma başlatmak için üç en iyi uygulamayı listeleyin.

**Eğitim verisindeki hedef:** Bir konuşma başlatmak için en iyi uygulamaların bazıları şunlardır: 1) Doğrudan olun ve açık uçlu bir soru sorun. 2) Sıcak ve dostça olun. 3) Gerçek ilgi göstererek övgüler sunun ve diğer kişinin ne söylediğini dinleyin.

**AMN'nin cevabı:** Bir konuşma başlatmak için en iyi uygulamaların bazıları şunlardır: 1) Doğrudan olun ve açık uçlu bir soru sorun. 2) Sıcak ve dostça olun. 3) Gerçek ilgi göstererek övgüler sunun ve diğer kişinin ne söylediğini dinleyin.

Ek tekrar: 160; önce NLL: 0.0033; sonra NLL: 0.0014

## Soru 2
**Girdi:** Web geliştirme ve web tasarımı arasındaki farkı açıklayın.

**Eğitim verisindeki hedef:** Web geliştirme, web sitesi oluşturma sürecidir ve web sitesinin işlevselliğini oluşturmak için kodlama ve işaretleme dili yazmayı içerir. Web tasarımı, web sitesinin görünümüne odaklanır ve görsel düzeni, renkleri ve tipografisini içerir. Web geliştirme, bir web sitesinin teknik yönlerine odaklanırken web tasarımı estetiğine odaklanır.

**AMN'nin cevabı:** Web geliştirme, web sitesi oluşturma sürecidir ve web sitesinin işlevselliğini oluşturmak için kodlama ve işaretleme dili yazmayı içerir. Web tasarımı, web sitesinin görünümüne odaklanır ve görsel düzeni, renkleri ve tipografisini içerir. Web geliştirme, bir web sitesinin teknik yönlerine odaklanırken web tasarımı estetiğine odaklanır.

Ek tekrar: 160; önce NLL: 0.0030; sonra NLL: 0.0013

## Soru 3
**Girdi:** Doğru bir çizgi için matematiksel denklemi türetin.

**Eğitim verisindeki hedef:** y = mx + b, burada m eğim ve b çizginin y-kesiti olarak tanımlanır.

**AMN'nin cevabı:** y = mx + b, burada m eğim ve b çizginin y-kesiti olarak tanımlanır.

Ek tekrar: 160; önce NLL: 0.0029; sonra NLL: 0.0015

## Soru 4
**Girdi:** Vejetaryenler için uygun bir yemek önerin.

**Eğitim verisindeki hedef:** Lezzetli bir vejetaryen yemek Ratatouille olabilir. Patlıcan, kabak, domates, biber, soğan, sarımsak ve otların zengin domates sosunda pişirilmesiyle yapılır. Quinoa, kuskus veya gevrek ekmekle servis edilebilir.

**AMN'nin cevabı:** Lezzetli bir vejetaryen yemek Ratatouille olabilir. Patlıcan, kabak, domates, biber, soğan, sarımsak ve otların zengin domates sosunda pişirilmesiyle yapılır. Quinoa, kuskus veya gevrek ekmekle servis edilebilir.

Ek tekrar: 160; önce NLL: 0.0032; sonra NLL: 0.0015

## Soru 5
**Girdi:** Dünyada kaç kuş türü olduğunu bulun.

**Eğitim verisindeki hedef:** Dünya'da yaklaşık 10.000 kuş türü vardır.

**AMN'nin cevabı:** Dünya'da yaklaşık 10.000 kuş türü vardır.

Ek tekrar: 160; önce NLL: 0.0028; sonra NLL: 0.0014

## Soru 6
**Girdi:** İyi dengelenmiş bir diyetin beş bileşenini adlandırın.

**Eğitim verisindeki hedef:** Dengeli bir diyet, meyve ve sebzeler; tam tahıllar; az yağlı süt ürünleri; sağlıklı yağlar ve az şeker, tuz ve sağlıksız yağ içeren besinler gibi çeşitli besin yoğunluğuna sahip yiyecekleri içermelidir. Ayrıca, dengeli bir diyet için yeterli su tüketimi ve gün boyunca düzenli öğünler ve atıştırmalıklar tüketmek önemlidir.

**AMN'nin cevabı:** Dengeli bir diyet, meyve ve sebzeler; tam tahıllar; az yağlı süt ürünleri; sağlıklı yağlar ve az şeker, tuz ve sağlıksız yağ içeren besinler gibi çeşitli besin yoğunluğuna sahip yiyecekleri içermelidir. Ayrıca, dengeli bir diyet için yeterli su tüketimi ve gün boyunca düzenli öğünler ve atıştırmalıklar tüketmek önemlidir.

Ek tekrar: 160; önce NLL: 0.0030; sonra NLL: 0.0014

## Soru 7
**Girdi:** Erkek ve dişi kardinal arasındaki farkı nasıl anlatacağınızı açıklayın.

**Eğitim verisindeki hedef:** Erkek kardinal kuşlar parlak kırmızı tüylere sahiptir, dişi kardinal kuşların tüyleri daha kahverengi renklidir. Erkeklerin yüzleri siyahken, dişi kardinal kuşların yüzleri gri renklidir. Ayrıca, erkek kardinal kuşlar genellikle dişilerden daha büyüktür.

**AMN'nin cevabı:** Erkek kardinal kuşlar parlak kırmızı tüylere sahiptir, dişi kardinal kuşların tüyleri daha kahverengi renklidir. Erkeklerin yüzleri siyahken, dişi kardinal kuşların yüzleri gri renklidir. Ayrıca, erkek kardinal kuşlar genellikle dişilerden daha büyüktür.

Ek tekrar: 160; önce NLL: 8.7594; sonra NLL: 0.0025

## Soru 8
**Girdi:** Bir resmi tanımlamak için 10 sıfat bulun.

**Eğitim verisindeki hedef:** Canlı, karmaşık, hipnotize edici, cezbedici, parlak, cesur, dinamik, ifade dolu, hipnotize edici ve baştan çıkarıcı.

**AMN'nin cevabı:** Canlı, karmaşık, hipnotize edici, cezbedici, parlak, cesur, dinamik, ifade dolu, hipnotize edici ve baştan çıkarıcı.

Ek tekrar: 160; önce NLL: 9.0933; sonra NLL: 0.0023
