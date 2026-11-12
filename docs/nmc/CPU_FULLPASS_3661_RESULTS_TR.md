# AMN-NMC CPU — 63 sayfalık tam ilk eğitim geçişi (12 Kasım 2026)

**Mimari hedef:** Kendi recurrent + fast-memory + episodic-memory mekanizmalarıyla ham metinden otoregresif dil öğrenen Transformer-dışı genel amaçlı nöral temel model. Bu koşu AGI'nin veya akıcı Türkçenin elde edildiğini göstermiyor.

## Kaynak ve bütünlük

- Eğitim kaynağı, kullanıcı tarafından seçilen **tek** akademik PDF; önceki Wikipedia korpusu bu eğitimde **kullanılmadı**.
- PDF SHA-256: `bc79c1ffdb3e78327496cff4d7e922d613d60275904a141bcf469e68cf5e1d20`.
- **63/63 sayfa**, **3.661/3.661 adım**, **232.186/232.186 hedef byte**; tam geçiş doğrulaması `one_full_epoch_complete=true`.
- Model: 93.642 parametre, 2 AMN blok; CPU FP32, seed 17.
- Çalışma, eldeki 900-adım checkpoint'inden tamamlandı; sonuçlar özel yerel çalışma alanında üretildi. Ham PDF ve özel korpus GitHub'a eklenmedi.

## Tam sonuçlar

| Ölçüm | Sonuç |
|---|---:|
| Son model (adım 3661) tam doğrulama BPB | **2,235165** / 28.394 hedef byte |
| Doğrulama alt bölmesinde seçilen en iyi checkpoint | adım **3375** |
| Seçilen checkpoint'in ayrılmış test BPB | **2,114611** / 29.324 hedef byte |
| Karşıt-örnekli bilgi/akıl yürütme denemesi (adım 3661) | **6/12** |
| İki yönü de doğru çözülen karşıt çift | **0/6** |
| Geçerli UTF-8 üretimi | test edilen 17/29/43 tohumlarında **evet** |
| Nöral oturum durumu geri yüklenince devam çıktıları | **birebir aynı** |

**Başarısızlık:** Üretilen Türkçe metinler hâlâ yer yer anlamsız ve dilbilgisel olarak bozuk. Karşıt olgular ve bileşik ilişki çıkarımları başarıyla öğrenilmiş değil. Bu yüzden genel akıl yürütme ve semantik uzun dönem bellek **kanıtlanmadı**.

## Sıralı öğrenme (ayrı küçük modeller)

A→B verisi, aynı PDF'nin eğitim sayfalarından alındı; 17/29/43 bağımsız tohumda, iki aşamada ayrı ayrı 1.536 byte işlendi. Üç tohumda B kaybı düşerken A kaybı da düşmüş durumda. Bu, aynı kaynak içindeki öğrenilebilirliği gösterir; farklı görev ve alanlar arasında katastrofik unutmanın çözüldüğünü göstermez. Bu küçük modellerin sonuçları resmi 93.642 parametreli checkpoint ile karıştırılmamalıdır.

## Bilimsel sınırlar ve açık mimari sorunlar

1. Tek akademik PDF, genel Türkçe dil dağılımını temsil etmez; validation/test aynı belgenin farklı sayfalarıdır.
2. Eğitim her sayfa başlangıcında recurrent durumu sıfırlıyor; uzun bağlam ve belge boyunca bellek sürekliliği için doğru değildir.
3. Yalnız bir büyük-model seed var; bellek ablasyonu, bağımsız veri kaynakları ve ölçek denemeleri eksiktir.
4. UTF-8 kodlama doğruluğu doğal dil yetkinliği anlamına gelmez.
5. Sonuçlar AGI, geniş kapsamlı akıl yürütme, ömür boyu öğrenme veya sınırsız uzun dönem bilgi saklama kanıtı değildir.

**Kod:** `src/amn_nmc`; **testler:** `tests/test_nmc_capability.py`, `tests/test_nmc_fullpass_continual.py`. GitHub Actions her ikisini içeren iş akışında başarılıdır. Eski korpus aktif geliştirme dalından çıkarılmıştır; diğer Git dalları ve geçmişteki kopyalara bu commit dokunmamıştır.
