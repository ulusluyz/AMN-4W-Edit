# AMN-4W-Edit — Eğitimi hızlandırma, matematiği koruma (2026-11-11)

**Karar:** Mevcut modeli veya checkpointleri değiştirmeden, ayrı deney modülünde önişlenmiş K/V/η projeksiyonları ve tamamen kapalı bellek yazma sütunlarını atlama yöntemleri eklendi. Kod: `experiments/safe_memory_speedup.py`; regresyon: `tests/test_speed_safe_memory.py`.

## Neden güvenli?

Orijinal sentetik proxy bellek güncellemesi:

\[
S_{t+1} = S_t + \eta_t (v_t - S_t k_t) k_t^T.
\]

Anahtarlar, değerler ve yazma kapısı token temsillerinden üretiliyor; geçmiş \(S\) durumuna bağlı değiller. Bu yüzden projeksiyonları zaman döngüsünün önüne taşımak, matematiği değiştirmeden büyük tensör çarpımlarını birleştirebilir. Ayrıca **batch'in tüm örneklerinde** \(\eta_t=0\) ise güncelleme özdeşliktir (\(S_{t+1}=S_t\)); yalnızca o sütun atlanır. Bir örnekte bile yazma etkinse adım atlanmaz.

Bu bir **hesaplama optimizasyonudur**, yeni bellek mimarisi değildir. Öğrenme maskesini, hedefleri, kayıp fonksiyonunu, parametreleri veya iki aşamalı sorguyu değiştirmez. Seçici silme/SVD dual-write testlerinin yerine geçmez.

## Gerçek CPU ölçümü

Ortam: PyTorch 2.10.0+cpu, bir CPU iş parçacığı. 50 token, 2 zincir, batch 64, 350 Adam adımı, aynı veri/seed, her seed 500 değerlendirme örneği. Model: yalnızca cevap kaybıyla eğitilmiş bağımsız AMN sentetik proxy; **production v0.4 değil**.

| Seed | Orijinal | Önprojeksiyon + kapalı sütun atlama | Süre azalması | Orijinal doğruluk | Optimize doğruluk |
|---|---:|---:|---:|---:|---:|
| 0 | 5,14 s | 3,17 s | %38,3 | %100 | %100 |
| 1 | 4,92 s | 3,37 s | %31,6 | %99,8 | %99,8 |
| 2 | 3,91 s | 3,83 s | %2,0 | %99,8 | %99,8 |

Üç seed'in ham süre ortalaması: orijinal **4,66 s**, optimize **3,46 s**, yaklaşık **%25,8** azalma. Süreler gürültülüdür, tek ölçümdür; kesin hızlanma veya 1000 adım hızlanması olarak genellenemez. İlk seed'de ısınma etkisi olabilir.

Başlangıçta eşit parametrelerle 5 seed üzerinde **kayıp farkı 0**, **bellek durumu farkı 0**, en yüksek gradyan farkı yaklaşık **2,04×10⁻¹⁰** ölçüldü (yalnızca toplu projeksiyon optimizasyonu). Eğitim sırasında float32 GEMM yuvarlama düzeni farklı olduğundan final loss çok az ayrışabilir. Doğruluk 3 seed için değişmedi.

## Koruma ve sonraki doğrulama

- Eski kod ve mevcut ağırlıklar **değiştirilmedi**.
- Yeni modül manuel test içindir; modelin eğitim koduna otomatik bağlanmadı.
- Yeni pytest testi kaynak model ile yeni yordamın forward/gradyan eşdeğerliğini ortak maskeli, örnek bazlı maskeli ve tamamen kapalı durumlarda denetler. Bu pytest dosyasının GitHub CI üzerinde başarılı tamamlandığı henüz **doğrulanmadı**.
- 1000 adım, 5 seed, daha uzun zincirler ve CPU/GPU karşılaştırması henüz yapılmadı.
- GPU'da maskeyi Python ile kontrol etmek senkronizasyon maliyeti getirebilir. Statik maskeli görevlerde etkin yazma adımları önceden belirlenmelidir.
- Hız kazanımı kaliteyi azaltmıyorsa ancak o zaman eğitim yoluna entegre edilebilir.

**Dondurulmuş `AMN-4W-jules` deposuna yazılmadı. Yalnızca `AMN-4W-Edit` güncellendi.**
