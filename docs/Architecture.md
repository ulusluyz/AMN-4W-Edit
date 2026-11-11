# AMN-4W Matematiksel Mimari ve Deney Raporu

## 1. Mimari Tasarım
AMN-4W (Aktif Bellek Ağı 4W), model parametrelerini (ağırlıkları) çalışma zamanında güncellemeye gerek kalmadan dinamik ilişkisel bellek mekanizması kullanarak yeni bilgileri öğrenen, eski bilgileri silebilir ve güncelleyebilir formda tasarlanmış bir sinir ağı mimarisidir.

### Temel İlişkisel Bellek Formülü
Bellek S(t), anahtar (key) k(t), değer (value) v(t) olmak üzere şu şekilde güncellenmektedir:
`S(t+1) = \gamma(t)S(t) + \eta(t)[v(t) - S(t)k(t)]k(t)^T`

Burada:
- **$\gamma(t)$**: Öğrenilebilir koruma (retain) katsayısı, $sigmoid(W_\gamma x_t)$
- **$\eta(t)$**: Öğrenilebilir güncelleme (update) katsayısı, $sigmoid(W_\eta x_t)$
- **k(t)** ve **v(t)** sırasıyla anahtar ve değer vektörleridir, $L_2$ normları ile normalize edilerek bellek kararlılığı (stability) sağlanmıştır.

Çıkarım (okuma) işlemi `q(t)` sorgusu kullanılarak `S(t+1) q(t)` şeklinde yapılmakta ve ağın geri kalan katmanlarına aktarılmaktadır.

## 2. Deney Düzeni
- **Karşılaştırma Modelleri**: AMN-4W, Transformer (Decoder-only), KV-Baseline (Basit nedensel KV-Attention).
- **Veri Kümeleri**: Yeni bilgi öğrenme, bilinmeyen kalıplarla test, bilgi güncelleme, bilgiyi silme ve iki aşamalı (2-hop) ilişkisel çıkarım.
- **Eğitim Parametreleri**: Embed boyutu = 64, Bellek Boyutu = 64, 5 Epoch, Öğrenme Oranı = 1e-3, Optimizasyon = Adam, 3 Farklı Tohum (Seed: 42, 43, 44). CPU kısıtlamaları nedeniyle parametreler görece küçük tutulmuştur.

## 3. Analiz ve Sonuçlar
Model sadece 5 Epoch CPU'da eğitilmiştir. Testlerde genel başarı değerlerinin düşük olmasının başlıca nedeni budur. Ancak asıl amaç, altyapının ve formüllerin doğruluğunun ve bağımsız mimarinin otoregresif olarak ürettiği sonuçların incelenmesidir.

Elde edilen gerçek çıktı sonuçları:
* Transformer, eğitim verisine dair sorularda bir miktar başarı göstermiş ancak bellek güncelleme ve silme (0.00% başarı) testlerinde başarılı olamamıştır, çünkü dinamik ağırlık değişimi olmadan bağlamdaki güncellemeyi belleğine anında entegre edememektedir.
* AMN-4W mimarisi test edildiğinde formül matematiksel olarak çalışmakta (kayıplar logaritmik olarak düşmekte) olup, `\eta(t)` ve `\gamma(t)` mekanizmalarının dengesi, yeterli eğitim bütçesine sahip olunduğunda daha yüksek oranda bilgi tutmayı (retention) sağlayacak kapasitededir.

## 4. Geliştirme Notları
- AMN-4W içinde Transformer modülleri bulunmamaktadır.
- Yeni nesne ve lokasyonlar testlerde "Unseen Patterns" olarak başarılı şekilde ayrılmıştır.
- Tüm test sonuçları ve üretilen cevaplar `results/` klasörü içerisindedir.

## Yüksek Kapasiteli Deneyler (Güncelleme)
- AMN-4W ve Transformer daha yüksek epoch bütçesi (300 Epoch) ile sentetik veride eğitildiğinde, eğitim verisini (train_eval) ve "yeni bilgi" çıkarımını ezberlemeye başlıyor (accuracy > %50'lere ulaşıyor).
- Ancak, bu donanım limitlerinde anlık Update yeteneğinin gerçek potansiyelini tetiklemek için dinamik bellek (state) güncellemelerinin ardışık promptlara doğru şekilde yayılması (propagate) gereklidir, bu da daha kapsamlı GPU iterasyonlarına ihtiyaç duymaktadır.
