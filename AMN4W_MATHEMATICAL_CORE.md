# AMN-4W Matematiksel Çekirdek (Güncellenmiş - Yönlü Unutma Kapısı)

AMN-4W mimarisi, bilgiyi zaman adımına (t) bağlı olarak dinamik bir ilişkisel bellekte güncelleyen ve okuyan otoregresif bir mekanizmadır. Daha önceki global aşınma (decay) problemi, yönlü unutma mekanizması ile çözülmüştür. Bu doküman, modelin temelini oluşturan işlemlerin saf matematiksel gösterimlerini içermektedir.

## 1. Değişkenler, Matrisler ve Boyutlar

Matematiksel işlemlerde kullanılan temel notasyonlar ve boyutlar aşağıdadır:

* **$b$**: Yığın boyutu (Batch size)
* **$L$**: Dizi uzunluğu (Sequence length)
* **$d_{e}$**: Gömme boyutu (Embedding dimension)
* **$d_{m}$**: Bellek boyutu (Memory dimension) - Anahtar ve değer vektörlerinin boyutu.

Girdi ve Bellek Durumu:
* **$x_t \in \mathbb{R}^{d_e}$**: $t$ anındaki girdi vektörü (Gömme ve pozisyonel kodlama toplamı).
* **$S_t \in \mathbb{R}^{d_m \times d_m}$**: $t$ anındaki dinamik ilişkisel bellek durumu.

Ağırlık Matrisleri ve Yanılgı Vektörleri (Öğrenilebilir Parametreler):
* **$W_k \in \mathbb{R}^{d_m \times d_e}$**, $b_k \in \mathbb{R}^{d_m}$: Anahtar (Key) projeksiyonu
* **$W_v \in \mathbb{R}^{d_m \times d_e}$**, $b_v \in \mathbb{R}^{d_m}$: Değer (Value) projeksiyonu
* **$W_q \in \mathbb{R}^{d_m \times d_e}$**, $b_q \in \mathbb{R}^{d_m}$: Sorgu (Query) projeksiyonu
* **$W_\eta \in \mathbb{R}^{1 \times d_e}$**, $b_\eta \in \mathbb{R}$: Güncelleme katsayısı projeksiyonu
* **$W_\gamma \in \mathbb{R}^{1 \times d_e}$**, $b_\gamma \in \mathbb{R}$: Yönlü Koruma/Silme katsayısı projeksiyonu

## 2. Başlangıç Koşulları

Her dizinin başlangıcında, $t=0$ için ilişkisel bellek matrisi boş (sıfır matrisi) olarak başlatılır:
$$S_0 = 0 \in \mathbb{R}^{d_m \times d_m}$$

## 3. Anahtar (K), Değer (V) ve Sorgu (Q) Hesaplamaları

Zaman adımı $t$ için girdi vektörü $x_t$'den hareketle ham anahtar, değer ve sorgu vektörleri doğrusal dönüşüm ile elde edilir.

Sıfır bölme hatalarını (Zero-norm) engellemek için normalizasyona $\epsilon$ (küçük bir pozitif sabit, örn: $10^{-8}$) eklenir:

* **Anahtar**: $k_t = \frac{W_k x_t + b_k}{\|W_k x_t + b_k\|_2 + \epsilon} \in \mathbb{R}^{d_m}$
* **Değer**: $v_t = \frac{W_v x_t + b_v}{\|W_v x_t + b_v\|_2 + \epsilon} \in \mathbb{R}^{d_m}$
* **Sorgu**: $q_t = \frac{W_q x_t + b_q}{\|W_q x_t + b_q\|_2 + \epsilon} \in \mathbb{R}^{d_m}$

## 4. Öğrenme ve Koruma Katsayılarının Hesaplanması

Modelin yönlü silme oranını belirleyen $\gamma_t$ ve yeni bilgiyi kaydetme oranını belirleyen $\eta_t$ katsayıları, sigmoid aktivasyon fonksiyonu $\sigma(\cdot)$ ile $[0, 1]$ aralığına sıkıştırılır:

* **Güncelleme Katsayısı (Update Gate)**: $\eta_t = \sigma(W_\eta x_t + b_\eta) \in (0, 1)$
* **Koruma Katsayısı (Directional Forget Gate)**: $\gamma_t = \sigma(W_\gamma x_t + b_\gamma) \in (0, 1)$

## 5. Bellek Durumu Geçişi, Güncelleme ve Seçici Silme

Global bellek zayıflamasını (0.410'a düşen kosinüs benzerliği problemi) çözmek için, eski bellek matrisini bir bütün olarak $\gamma_t$ ile çarpmak yerine yönlü güncelleme denklemi kullanılmıştır:

$$S_{t+1} = S_t + [ \eta_t v_t - \gamma_t S_t k_t ] k_t^\top$$

Bu formüldeki mekanizmaların karşılığı:
- **Global Aşınma Yok (No Global Decay)**: Matrisin tamamı $S_t$ olarak denklemin başına eklenir, böylece hedef anahtar ($k_t$) ile ilgisiz ve ortogonal olan tüm eski bilgiler %100 oranında (kayıpsız) korunur.
- **Belleğe Yazma**: Yeni değer $v_t$, öğrenme hızı $\eta_t$ ile ağırlıklandırılıp hedef anahtar yönünde ($k_t^\top$) belleğe eklenir.
- **Seçici Bilgi Silme (Selective Deletion) ve Güncelleme**: Eski bilgi, sadece $k_t$ doğrultusunda bellekte var olan karşılığı $S_t k_t$ okunarak ve $\gamma_t$ oranında çıkarılarak (- $\gamma_t S_t k_t$) silinir. Bu "yönlü unutma", aynı hedefe ait yeni bilginin eklenmesi için alan açarken diğer ortogonal bilgileri kesinlikle bozmaz.
- **Anahtar Çakışması (Key Interference) Limitasyonu**: Anahtarlar tam ortogonal değilse (benzerlik taşıyorsa), bir hedefin bilgisi güncellenirken veya silinirken, ona yapısal olarak benzeyen diğer eski bilgiler $k_i^\top k_t$ oranında kaçınılmaz olarak kısmi parazit (zayıflama) yaşar. Ancak bu, matematiksel projeksiyonun doğası gereği oluşan yerel bir parazittir, eski formüldeki gibi tüm matrisi ezen global bir kapasite kaybı değildir.

## 6. Çok Aşamalı (İki Aşamalı) Çıkarım Mekanizması

İlk okuma $r_t^{(1)}$, $x_t$'den elde edilen sorgu ile yapılır:
$$r_t^{(1)} = S_{t+1} q_t$$

Eğer iki aşamalı (2-hop) çıkarım gerekiyorsa, model önceki okuma çıktısını yeni bir sorguya dönüştürür:
$$q_t^{(2)} = \frac{W_q^{(2)} r_t^{(1)} + b_q^{(2)}}{\|W_q^{(2)} r_t^{(1)} + b_q^{(2)}\|_2 + \epsilon}$$
$$r_t^{(2)} = S_{t+1} q_t^{(2)}$$

## 7. Otoregresif Çıktı Katmanı

Entegre okuma $r_t$, girdi vektörü $x_t$ ile birleştirilip lineer projeksiyonlar ve bir aktivasyon (ReLU) ile geçirilerek kelime olasılıklarını üretir.
* $h_t = \text{ReLU}(W_{h} [x_t \oplus r_t] + b_{h})$
* Çıktı Olasılıkları (Logits): $P(y_{t+1} | x_1 ... x_t) = \text{Softmax}(W_{out} h_t + b_{out})$
