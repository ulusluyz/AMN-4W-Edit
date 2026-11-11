# AMN-4W-Edit: Yapılandırılmış Türkçe ilişkisel öğrenme ve dual bellek deneyi

Tarih: 11 Kasım 2026
Durum: Yerel ölçüm ve GitHub Actions üzerinde yeniden üretim tamamlandı. Eski eğitim motoru ve eski model ağırlıkları değiştirilmedi. AMN-4W-jules deposu salt okunur tutuldu.

## Kapsam

Deney sıfırdan eğitilen üç küçük modeli karşılaştırır: standart delta-bellekli AMN, çarpımsal kişi–ilişki anahtarı ve Gram sistemiyle korunan dual AMN, iki katmanlı Transformer.

Metin örneği:

  Ayşe konum okul. Can renk kırmızı. Ali renk mavi. Sena konum garaj.
  Soru: Ali renk?  -> mavi

Buradaki metinler otomatik üçlü dilbilgisine sahiptir. Model girdisi, üçlü sınırları ile kişi/ilişki/değer alanları ayrılmış token ID'leridir. Bu ayrıştırma çıkarımda hazır olarak sağlanır; serbest Türkçe anlama veya üretme öğrenilmiş değildir. Altın cevap çıkarıma verilmez; cevap, bağlamdaki değerlerden yalnız cevap çapraz entropisiyle öğrenilir.

## Veri

20 kişi adı, iki ilişki (renk ve konum), her biri sekiz değer: toplam 320 olası kişi–ilişki–değer eşleşmesi. SHA-256'ya göre bölme:

- Eğitim: 264 üçlü.
- Doğrulama: 25 üçlü.
- Test: 31 üçlü.

Bölmelerde ortak üçlü yoktur. Fakat bütün kelimeler eğitimde görülür. Her örnek 4 olgudan ve bir sorudan oluşur. Bir bağlamda aynı kişi+ilişki anahtarı iki kez bulunmaz. 15.000 farklı örnek, 15.000 farklı ilişki değil; 31 test eşleşmesi etrafında örneklenen farklı bağlamlardır. Deney geliştirmesi sırasında test bölümüne bakılmıştır: bu nedenle test eğitimden ayrık olsa da tamamen kör nihai sınama değildir.

## Matematik ve kritik düzeltme

İlk lineer kişi+ilişki anahtarı, iki kişi ve iki ilişki kombinasyonlarında doğrusal bağımlılık oluşturdu. Sayısal koruma testi başarısız oldu. Bu sorun düzeltilerek, kişi ve ilişki gömmelerinin eleman bazında çarpımıyla öğrenilen anahtar kullanıldı:

  k(a,r) = normalize( W( e(a) elementwise_times e(r) ) + b )

Standart delta-AMN belleği:

  S_next = S + (v - S k) k^T

Dual AMN, K (anahtar satırları), V (değer satırları) ve küçük ridge ile:

  D = solve(K K^T + 1e-5 I, K)
  S = V^T D
  read(q) = S q

Tam rank ve sıfır ridge durumunda dual anahtarlar diğer adreslerin okumalarını bozmaz. Gerçekte ridge sıfır değildir; yakın/çakışan anahtarlarda tam koruma garantisi yoktur. Dual model K'yi saklar ve Gram çözümü yapar; dolayısıyla sabit boyutlu AMN belleğiyle aynı hesaplama/bellek ölçeğine sahip değildir. Ayrı model olarak korunmuştur.

## GitHub üzerinde yeniden üretilen 50 adımlık sonuçlar

Ortam: GitHub Actions ubuntu-latest CPU; PyTorch sürümü sonuçlardaki metadata.json içindedir. Tek PyTorch iş parçacığı. Adam lr=0,003; batch=128; boyut 48; aynı eğitim örneği dizisi; seed=0,1,2,3,4. Her seed için 1.000 doğrulama ve 3.000 test soru-cevap örneği. Beş seed için toplam 15.000 test sorusu.

| Model | Parametre | Adım | Doğru / 15.000 | Tam cevap doğruluğu | Ortalama eğitim süresi / seed | Ortalama test CE |
|---|---:|---:|---:|---:|---:|---:|
| Standart delta-AMN | 8.874 | 50 | 14.996 | %99,973 | 0,657 sn | 0,037996 |
| Dual AMN | 8.874 | 50 | 15.000 | %100,000 | 0,349 sn | 0,030812 |
| Transformer | 42.666 | 50 | 7.000 | %46,667 | 1,094 sn | 1,355079 |

Aynı 50-adım bütçesinde dual AMN, delta-AMN'den yaklaşık 1,88 kat; bu Transformer'dan 3,13 kat kısa sürede eğitildi. Bu, donanımdan bağımsız bir hız veya genel mimari üstünlüğü değildir.

Standart delta-AMN'nin erken durdurmalı ayrı yerel koşularında 5 seed'den 4'ü 50 adımda, biri 100 adımda iki ardışık %100 dev kontrolünü geçti. Hepsi 3.000/3.000 testte başarılıydı. Dual AMN her seed'de 25. adımdan itibaren %100 dev doğruluğuna ulaştı; iki ardışık kontrol şartıyla 50. adımda durdu.

## Bağlam uzunluğu stres testi (GitHub)

Eğitimde sadece 4 olgu kullanılmışken, yeni bağlamlarda 5 seed × 1.000 test:

| Olgu sayısı | Delta-AMN | Dual AMN |
|---:|---:|---:|
| 4 | 4.999 / 5.000 | 5.000 / 5.000 |
| 6 | 4.994 / 5.000 | 5.000 / 5.000 |
| 8 | 4.990 / 5.000 | 5.000 / 5.000 |
| 12 | 4.919 / 5.000 | 5.000 / 5.000 |

Stres örnekleri de aynı sınırlı sözcük ve ilişki uzayından seçildi. Uzun vadeli ve çok büyük ölçekli bellek dayanıklılığına kanıt değildir.

## Tekrar üretim / kaydedilmiş çıktılar

- Eğitim ve test: experiments/relational_qa.py
- Regresyonlar: tests/test_relational_qa.py
- Otomatik eğitim arşivleme: .github/workflows/relational_qa_archive.yml
- Bütün 15 koşunun JSON kayıtları ve 15 PyTorch .pt ağırlığı: results/relational_qa/equal50/
- Stres testleri: results/relational_qa/stress.jsonl
- Kaynak SHA-256, PyTorch sürümü ve veri bölmeleri: results/relational_qa/metadata.json

Komutlar (depo kökünden):

~~~bash
python -m pytest -q tests/test_relational_qa.py
python experiments/relational_qa.py --model all --seeds 0 1 2 3 4 --steps 50 --batch 128 --dev-every 25 --dev-size 1000 --test-size 3000
~~~

Regresyon testleri yerelde geçti; GitHub Actions çalışma kaydı da başarılı. Model ağırlıkları kaynak arşivde saklandı. Eski AMN-4W Edit modelini veya önceki deneyleri otomatik olarak değiştirmez.

## Henüz başarılmayan hedefler

Bu deneyde görülen %100, sadece sabit dilbilgili kişi–renk/konum tek kelimelik ilişkisel QA için gözlenen doğruluktur. Serbest Türkçe soruları ayrıştırma, yeni sözcükleri anlama, serbest uzun cevap üretimi, bir anahtarın zaman içindeki güncellenmesi/silinmesi, iki aşamalı sorgular ve genel dil modelleme henüz çözülmedi. Yeni model gerçek bir genel LLM veya Transformer'ın bütün görevlerde yerini alan bir sistem olarak değerlendirilmemelidir.
