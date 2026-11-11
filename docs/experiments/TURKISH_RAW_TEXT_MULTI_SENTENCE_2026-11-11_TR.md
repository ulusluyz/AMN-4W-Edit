# AMN-4W-Edit — Türkçe ham metinden çok cümleli sinir ağı yanıtı

Tarih: 11 Kasım 2026. Çalışma deposu: **AMN-4W-Edit**. Dondurulmuş **AMN-4W-jules** deposuna yazılmadı.

## Amaç ve gerçek kapsam

Yeni prototip üç ayrı, doğrulanabilir katman kullanır:

1. **Unicode Türkçe metin:** Kelimelere bölme işlemi sözdizimsel bir ayırmadır; kural tabanlı kişi/özellik/değer ayrıştırması kullanılmaz. Eğitilmiş sözcük-etiketleme ağı kişi, ilişki ve değeri tespit eder.
2. **Dual Upsert AMN:** Önceden eğitilmiş ve dondurulmuş 50-adımlık orijinal Dual checkpoint'i bağımsız ilişkileri okur; daha önce doğrulanan son-yazma önceliğini sağlar.
3. **Sinirsel cümle üretici:** GRU, <bos> tokeninden başlayıp önceki tahminleriyle her kelimeyi ve <eos> tokenini üretir. Çıkarım sırasında doğru cümle veya cevap şablonu parametre olarak verilmez.

**Çok önemli sınırlama:** Eğitim verisi, 20 kişi, iki özellik (konum ve kalem rengi), her özellik için sekiz değer ve on eğitim ifade biçiminden otomatik oluşturulmuş *sonlu bir Türkçe sentetik görevdir*. İki farklı ifade biçimi eğitimde gizli tutuldu. Çıktı hedefleri de üç elle yazılmış Türkçe yanıt tarzından türetilmiştir. Bu sonuç, **herhangi bir konuda serbest Türkçe anlama, sınırsız uzunlukta özgün kompozisyon ya da genel LLM eğitiminin tamamlandığı** anlamına gelmez.

## Donanım ve eğitim

GitHub Actions CPU, PyTorch 2.15.0, tek PyTorch iş parçacığı. Önceki Dual AMN **8.874** parametre, değişmeden kullanıldı. Yeni Türkçe etiketleyici yaklaşık **25.192**, yeni otomatik kelime üreteci yaklaşık **96.512** öğrenilebilir parametreye sahip.

- Üçlü bilgilerin SHA-256 tabanlı bölünmesi: eğitim **264**, geliştirme **25**, test **31** farklı bağ.
- Sözcükler biliniyor; testte yeni kişi veya kelime değil, yeni kişi–özellik–değer eşleşmeleri var.
- Girişte her örnekte **4 ayrı bilgi cümlesi**. Nokta sınırlarıyla ayırma uygulanır; iki bilginin tek cümlede harmanlanması çözülmüş değildir.
- Üretici: **1.600 adım**, **29,74 saniye**; tek başına held-out tam cümle **1.181/1.200** (%98,42).
- Yeni öğrenilmiş ayrıştırıcı geliştirmede **50. adımda %100**, eğitimde görülmemiş bağ + bilinen ifadelerde **1.200/1.200**, daha önce hiç kullanılmayan iki ifade kalıbında **1.200/1.200** tam bilgi çıkarımı.
- Uçtan uca cevap testi: modelin kendisine **yalnız ham Türkçe girdi ve soru verilir**; altın üçlü veya altın cevap verilmez. Yanlış ve reddedilen yanıtlar toplam doğruluk paydasındadır.

## Bağımsız uçtan uca sonuçlar (GitHub'daki ağırlıklar)

| Test ailesi | Toplam | Kabul | Tam doğru uzun cevap | Doğruluk |
|---|---:|---:|---:|---:|
| Eğitimde kullanılan ifade türleriyle **testteki yeni bilgi bağları** | 600 | 600 | **593** | **%98,83** |
| Eğitimde görülmemiş iki ifade türüyle **testteki yeni bilgi bağları** | 600 | 600 | **584** | **%97,33** |

Test sonuçları bağımsız *ifade+bilgi kombinasyonları* içerir; tümü sadece 31 test üçlüsünden türetilmiştir. Başarılar gerçek dünya dağılımına genelleme garantisi değildir. Ayrıntılı cümlelerde bir önceki kelime tahminindeki hata sonraki kelimelere yayılabilmektedir. Kısa ve açıklayıcı biçimler ayrıntılı biçimden daha kararlıdır.

**Alan dışı dört deneme** (kara delik, yeni isim Bora/telefon, Ali neden sinirli, yağmur) reddedildi. Bu, dört elle seçilmiş negatif senaryo olup genel alan-dışı güvenilirliği göstermez.

## Modelin ürettiği örnek — eğitimde olmayan ifade biçimi

**Girdi:**

Deniz, beyaz renginde bir kaleme sahip. Deniz, okulda zaman geçiriyor. Murat, sarı renginde bir kaleme sahip. Selin, garajda zaman geçiriyor.

**Türkçe soru:** Deniz hangi renk ile ilişkilendiriliyor?

**Çıktı (kaydedilmiş gerçek sinir ağı tahmini):**

> Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Deniz'in kaleminin rengi beyazdır. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Deniz için belirtilen renk beyaz olarak görünüyor.

Bu 4 cümleli yanıt, çıktı tokenleri üzerinden otomatik üretildi. Ancak eğitimde benzer cümle biçimleri kullanıldı.

## Kaydedilmiş hata — görünürlüğü korunuyor

**Soru:** Yiğit'in kullandığı kalemi tarif eder misin?

**Beklenen son değer:** sarı.

**Model hatası:** İlk bilgi cümlesinde rengi doğru (sarı) söylemesine rağmen son cümlede rengi **mor** olarak tekrarladı. Buradaki hata bilgi çıkarımında değil, uzun otomatik üretimdeki tutarsızlıktadır. Bu yüzden %100 tam yanıt hedefi henüz sağlanmadı.

## Depoda bulunan kanıt ve kodlar

- **Kaynak:** `experiments/free_turkish_core.py`, `experiments/turkish_tagged_parser.py`, `experiments/evaluate_free_turkish.py`.
- **Regresyon:** `tests/test_free_turkish_core.py`.
- **Yeniden üretim:** `.github/workflows/free_turkish_generation.yml`.
- **Eğitilmiş model:** `results/turkish_tagged_v1/parser.pt`, `results/turkish_tagged_v1/decoder.pt`.
- **Ham sonuç:** `results/turkish_tagged_v1/end_to_end_validation.json`.
- **Gerçek cümle örnekleri:** `results/turkish_tagged_v1/end_to_end_examples_tr.md`.
- **Denetim:** `results/turkish_tagged_v1/SOURCE_SHA256.txt`.

Komut (depo kökünde):

~~~bash
python experiments/turkish_tagged_parser.py ask \
  --output results/turkish_tagged_v1 \
  --context "Deniz, beyaz renginde bir kaleme sahip. Deniz, okulda zaman geçiriyor. Murat, sarı renginde bir kaleme sahip. Selin, garajda zaman geçiriyor." \
  --question "Deniz hangi renk ile ilişkilendiriliyor?" \
  --style detayli --threshold 0.55
~~~

Stil seçenekleri: `kisa`, `aciklayici`, `detayli`.

**Karar:** Sinirsel ham Türkçe giriş + çok cümleli cevap üreten bir *sınırlı alan prototipi* gerçekleştirildi. Genel serbest Türkçe ve özgün, sınırsız konulu uzun metin üretimi, bu az veri ve küçük modelle doğrulanmadı. Mevcut AMN ve Transformer checkpoint'leri bozulmadı.
