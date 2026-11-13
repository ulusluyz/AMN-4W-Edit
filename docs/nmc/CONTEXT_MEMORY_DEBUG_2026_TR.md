# AMN-NMC — Bellek adresleme ve akıl yürütme hata analizi (13 Kasım 2026)

## Doğrulanmış sorun

Kullanıcının 93.642 parametreli seçilmiş checkpoint'i, aynı iki akademik PDF'den yalnız otoregresif **sonraki UTF-8 baytı** tahminiyle eğitildi. Bu hedef, bağlamdan karşıt bilgi ayırmayı veya çok adımlı zincir kurmayı doğrudan öğretmez. Eğitim 64-bayt TBPTT ile daha önceki bağlam için doğrudan gradyan keser.

Eski checkpoint'in ilk bellek blokunda sekiz slotun ortalama anahtar benzerliği **0,871**; normalize adresleme entropisi **0,974/1** idi. İlk çalışmada 12 doğal karşıt bilgi/akıl yürütme sorusunda 6 doğru; **0/6 iki yönlü karşıt çift**. Görev bağlamını değil popüler cevabı tercih etme eğilimi ölçüldü.

## Kod düzeltmesi

`NMCConfig.addressing` artık **legacy** (varsayılan) veya **competitive** olabilir. Competitive adresleme, anahtar/sorgu üzerindeki ortak pozitif bileşeni çıkarır ve belleğe seyrek yerleştirme uygular. Mevcut checkpoint'in ağırlık isimleri, şekilleri ve **legacy** çıkarım davranışı korunur. Eski NMC v2 checkpoint formatında eksik `addressing` değeri resume sırasında `legacy` kabul edilir.

Kodlama/UTF-8 sorunu farklıdır; geçerli UTF-8 üretimi tek başına anlamlı Türkçe veya akıl yürütme değildir.

## Gerçek deney: eski checkpoint değişmedi

Kullanıcının *yalnız iki PDF'sindeki kelimelerden türetilen tanımlayıcılar* ile rastgele karşıt ilişkiler üretildi. Yanıt-bölümü maskeli çapraz-entropy, yanlış yanıta karşı ranking loss, eski PDF replay ve öğretmen dağılımı koruması denendi. Kısa→uzun sembolik müfredatta 700 güncelleme yapıldı. Yeni dış kaynak veya gerçek-dünya ilişkisi öğretilmedi.

| Ölçüm | Eski | 700 adım doğrudan | Seçilmiş %75 yeni + %25 eski |
|---|---:|---:|---:|
| Eski PDF tam doğrulama BPB | 2,098704 | 2,143935 | **2,106576** |
| Eski PDF ayrılmış test BPB | 1,985640 | 2,038859 | **2,000568** |
| Kısa karşıt ilişki çiftleri, doğrulama | 0/18 | 11/18 | **10/18** |
| Yeni tanımlayıcılı son test, 27 çift | 0/27 | değerlendirilmedi | **7/27** |
| Son testte iki aşamalı ilişki | 0/9 | değerlendirilmedi | **0/9** |
| Doğal dil soruları | 6/12 | 6/12 | **6/12** |

Ağırlık karışımı **yalnız doğrulama** üzerinden seçildi. Eski PDF testi ardından değerlendirildi. Yeni tanımlayıcı testi seçimin ardından yapıldı. Çok adımlı doğal akıl yürütme ve akıcı Türkçe **başarılı değildir**.

## Karar

**Mevcut varsayılan model korunacak.** İlişki başarısı kazanan deneysel checkpoint ayrı verilir. Kod düzeltmesi bir regresyon yaratmamak için `legacy` varsayılanını korur; yalnız doğrulanmış çalışma modunda `competitive` seçilir.

Eğitim betikleri, özel checkpoint ve sayısal deney kayıtları kullanıcıya özel ZIP paketindedir. Korpusun kendisi herkese açık GitHub'a yüklenmedi. Yalnız iki akademik metinle genel bilgi ve akıcı Türkçe üretim için yeterli kanıt elde edilemez.
