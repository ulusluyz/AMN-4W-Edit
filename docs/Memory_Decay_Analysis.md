# Bellek Kaybı (Memory Decay) ve Çözüm Analizi

## Tespit Edilen Sorun
Önceki formülde yer alan `S(t+1) = \gamma_t S_t + ...` denklemindeki $\gamma_t$ (koruma) katsayısı, matrisi her işlemde skaler olarak küçültmektedir. Yeni ve alakasız bilgiler öğrenilirken bile $\gamma_t$ 1.0'ın altında kaldığı için, eski bilgiler global bir şekilde zayıflamaya başlar. Rastgele anahtarlarla yapılan 20 işlemlik bir deneyde (Seed 42), hedef bilginin ilk okuma büyüklüğüne (magnitude) oranı **0.409**'a kadar düşmüştür (belirtilen 0.410 sorununu tam olarak doğrulamaktadır). Anahtarlar tamamen ortogonal olsa bile sinyal büyüklüğü %64 oranında kaybedilmektedir.

## Geliştirilen Matematiksel Çözüm
Global bir bozunma (decay) yerine, silme ve güncelleme işlemlerinin **sadece hedeflenen anahtar (k_t) yönünde** yapılmasını sağlayan yeni denklem tasarlanmıştır:

**Yeni Denklem:**
$$S_{t+1} = S_t + \left[ \eta_t v_t - \big( (1-\gamma_t) + \eta_t \gamma_t \big) S_t k_t \right] k_t^\top$$

Bu denklemde $S_t$'nin geri kalanı korunurken, yalnızca $k_t$ doğrultusundaki bilgi modifiye edilir.

## Deney Sonuçları (20 İşlem Sonrası Bellek Koruma)

Geliştirilen `measure_memory_decay.py` betiği üzerinden 3 farklı tohumla (Seed 42, 43, 44) ölçümler yapılmıştır.

1. **Eski Formül (Global Gamma):**
   * **Ortogonal Anahtarlar:** Sinyal Büyüklüğü Korunumu: **0.358** (Tüm tohumlarda sinyal eziliyor)
   * **Rastgele Anahtarlar:** Sinyal Büyüklüğü Korunumu: Ortalama ~**0.41** (Benzerlik dalgalanıyor)

2. **Yeni Formül (Yönlü Koruma):**
   * **Ortogonal Anahtarlar:** Sinyal Büyüklüğü Korunumu: **1.000** (Kayıpsız tam koruma)
   * **Rastgele Anahtarlar:** Sinyal Büyüklüğü Korunumu: **0.887, 0.837, 0.912** (Kararlı)
   * **Benzer Anahtarlar:** Sinyal Büyüklüğü Korunumu: **0.868, 0.802, 0.828** (Kararlı)

Bu çözüm sayesinde bellek kapasitesi, alakasız bilgi aktarımlarından kaynaklı silinmelere karşı bağışık hale getirilmiş ve yalnızca hedeflenen güncelleme/silme işlemleri güvenle yapılabilir olmuştur. Anahtar çakışması (key interference) durumunda minimal kayıp yaşansa da, sinyalin genel büyüklüğünün kaybolma problemi çözülmüştür.
