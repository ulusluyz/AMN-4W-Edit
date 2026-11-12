# AMN v3.1 tekrarlı öğrenme — gerçek kaynak sorularıyla kontrol

Modelin daha önce SFT aşamasında gördüğü ve gerçek Türkçe veri kümesinde bulunan
altı soruya kontrollü ağırlık güncellemesi yapıldı. İki kontrol sorusu yeni
öğretimde kullanılmadı. Bu test **ezberleme / optimizasyon** denetimidir;
genel Türkçe anlama kanıtı değildir.

Orijinal checkpoint salt okunur kaldı. Kaynak SHA ve eğitim örneği kimliği
doğrulandı. Çıktılar otomatik olarak üretildi, elle düzeltilmedi.

Önce eğitim sorularında tam eşleşme: 0/6
Sonra eğitim sorularında tam eşleşme: 6/6
Ek eğitim adımı: 360

## Öğretilen soru 1
**Soru:** Bir konuşma başlatmak için üç en iyi uygulamayı listeleyin.

**Gerçek kaynak cevabı:** Bir konuşma başlatmak için en iyi uygulamaların bazıları şunlardır: 1) Doğrudan olun ve açık uçlu bir soru sorun. 2) Sıcak ve dostça olun. 3) Gerçek ilgi göstererek övgüler sunun ve diğer kişinin ne söylediğini dinleyin.

**AMN cevabı:** Bir konuşma başlatmak için en iyi uygulamaların bazıları şunlardır: 1) Doğrudan olun ve açık uçlu bir soru sorun. 2) Sıcak ve dostça olun. 3) Gerçek ilgi göstererek övgüler sunun ve diğer kişinin ne söylediğini dinleyin.

Eğitimde tekrar gösterim: 180
Önceki NLL: 6.8612; sonraki NLL: 0.0033

## Öğretilen soru 2
**Soru:** Web geliştirme ve web tasarımı arasındaki farkı açıklayın.

**Gerçek kaynak cevabı:** Web geliştirme, web sitesi oluşturma sürecidir ve web sitesinin işlevselliğini oluşturmak için kodlama ve işaretleme dili yazmayı içerir. Web tasarımı, web sitesinin görünümüne odaklanır ve görsel düzeni, renkleri ve tipografisini içerir. Web geliştirme, bir web sitesinin teknik yönlerine odaklanırken web tasarımı estetiğine odaklanır.

**AMN cevabı:** Web geliştirme, web sitesi oluşturma sürecidir ve web sitesinin işlevselliğini oluşturmak için kodlama ve işaretleme dili yazmayı içerir. Web tasarımı, web sitesinin görünümüne odaklanır ve görsel düzeni, renkleri ve tipografisini içerir. Web geliştirme, bir web sitesinin teknik yönlerine odaklanırken web tasarımı estetiğine odaklanır.

Eğitimde tekrar gösterim: 180
Önceki NLL: 7.1057; sonraki NLL: 0.0030

## Öğretilen soru 3
**Soru:** Doğru bir çizgi için matematiksel denklemi türetin.

**Gerçek kaynak cevabı:** y = mx + b, burada m eğim ve b çizginin y-kesiti olarak tanımlanır.

**AMN cevabı:** y = mx + b, burada m eğim ve b çizginin y-kesiti olarak tanımlanır.

Eğitimde tekrar gösterim: 180
Önceki NLL: 6.7892; sonraki NLL: 0.0029

## Öğretilen soru 4
**Soru:** Vejetaryenler için uygun bir yemek önerin.

**Gerçek kaynak cevabı:** Lezzetli bir vejetaryen yemek Ratatouille olabilir. Patlıcan, kabak, domates, biber, soğan, sarımsak ve otların zengin domates sosunda pişirilmesiyle yapılır. Quinoa, kuskus veya gevrek ekmekle servis edilebilir.

**AMN cevabı:** Lezzetli bir vejetaryen yemek Ratatouille olabilir. Patlıcan, kabak, domates, biber, soğan, sarımsak ve otların zengin domates sosunda pişirilmesiyle yapılır. Quinoa, kuskus veya gevrek ekmekle servis edilebilir.

Eğitimde tekrar gösterim: 180
Önceki NLL: 6.8491; sonraki NLL: 0.0032

## Öğretilen soru 5
**Soru:** Dünyada kaç kuş türü olduğunu bulun.

**Gerçek kaynak cevabı:** Dünya'da yaklaşık 10.000 kuş türü vardır.

**AMN cevabı:** Dünya'da yaklaşık 10.000 kuş türü vardır.

Eğitimde tekrar gösterim: 180
Önceki NLL: 6.2776; sonraki NLL: 0.0028

## Öğretilen soru 6
**Soru:** İyi dengelenmiş bir diyetin beş bileşenini adlandırın.

**Gerçek kaynak cevabı:** Dengeli bir diyet, meyve ve sebzeler; tam tahıllar; az yağlı süt ürünleri; sağlıklı yağlar ve az şeker, tuz ve sağlıksız yağ içeren besinler gibi çeşitli besin yoğunluğuna sahip yiyecekleri içermelidir. Ayrıca, dengeli bir diyet için yeterli su tüketimi ve gün boyunca düzenli öğünler ve atıştırmalıklar tüketmek önemlidir.

**AMN cevabı:** Dengeli bir diyet, meyve ve sebzeler; tam tahıllar; az yağlı süt ürünleri; sağlıklı yağlar ve az şeker, tuz ve sağlıksız yağ içeren besinler gibi çeşitli besin yoğunluğuna sahip yiyecekleri içermelidir. Ayrıca, dengeli bir diyet için yeterli su tüketimi ve gün boyunca düzenli öğünler ve atıştırmalıklar tüketmek önemlidir.

Eğitimde tekrar gösterim: 180
Önceki NLL: 6.7808; sonraki NLL: 0.0030

## Ek eğitimde kullanılmayan kontrol 1
**Soru:** Erkek ve dişi kardinal arasındaki farkı nasıl anlatacağınızı açıklayın.

**Kaynak cevabı:** Erkek kardinal kuşlar parlak kırmızı tüylere sahiptir, dişi kardinal kuşların tüyleri daha kahverengi renklidir. Erkeklerin yüzleri siyahken, dişi kardinal kuşların yüzleri gri renklidir. Ayrıca, erkek kardinal kuşlar genellikle dişilerden daha büyüktür.

**AMN cevabı:** Web geliştirme, web sitesi oluşturma sürecidir ve web sitesinin işlevselliğini oluşturmak için kodlama ve işaretleme dili yazmayı içerir. Web tasarımı, web sitesinin görünümüne odaklanır ve görsel düzeni, renkleri ve tipografisini içerir. Web geliştirme, bir web sitesinin teknik yönlerine odaklanırken web tasarımı estetiğine odaklanır.

Önceki NLL: 7.0695; sonraki NLL: 8.7594

## Ek eğitimde kullanılmayan kontrol 2
**Soru:** Bir resmi tanımlamak için 10 sıfat bulun.

**Kaynak cevabı:** Canlı, karmaşık, hipnotize edici, cezbedici, parlak, cesur, dinamik, ifade dolu, hipnotize edici ve baştan çıkarıcı.

**AMN cevabı:** Bir konuşma başlatmak için en iyi uygulamaların bazıları şunlardır: 1) Doğrudan olun ve açık uçlu bir soru sorun. 2) Sıcak ve dostça olun. 3) Gerçek ilgi göstererek övgüler sunun ve diğer kişinin ne söylediğini dinleyin.

Önceki NLL: 6.4169; sonraki NLL: 9.0933
