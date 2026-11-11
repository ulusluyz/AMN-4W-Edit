# AMN-4W Edit — %100 doğruluk ve minimum eğitim süresi denetimi

**Tarih:** 11 Kasım 2026  
**Çalışma deposu:** `ulusluyz/AMN-4W-Edit`  
**Korunan depo:** `AMN-4W-jules` (salt okunur)  
**Durum:** Eğitim verisi üst sınırı ve erken durdurma ölçüldü; **yeni sorularda %100 hedefi sağlanmadı**.

## 1. Önce doğruluk hedefinin sınırı

Mevcut `data/train.json` dosyasında **6** Türkçe soru-cevap kaydı var. Causal next-character eğitimi için 438 pad-dışı hedef karakterin bazıları, **tamamen aynı geçmiş diziyi** izlemelerine karşın farklı bir sonraki karakter istiyor. Bu çelişkiler hiçbir deterministik nedensel tahmincinin bütün eğitim tokenlarını %100 doğru sınıflandırmasına izin vermez.

```
Toplam hedef token:             438
Ulaşılabilir maksimum doğru:    431
Kaçınılmaz hata:                  7
Maksimum train token doğruluğu: %98,401826484
Minimum teorik CE kaybı:         0,0245446503 nat/token
Çelişen nedensel önek:           5
```

Örnek: `Aliin kalemi ...` önekinden sonra aynı kayıtlarda hem `mavidir` hem `kırmızıdir` devamı bulunuyor. Başka çelişkiler de kayıt başında ve aynı bağlamı izleyen iki farklı soru tipinde görülüyor.

Bu sınır, **tüm metnin bir sonraki karakter tahmini** içindir. **Soruların yalnızca cevaplarını %100 doğru üretmek** ayrı bir hedeftir ve bu matematiksel kısıt onu doğrudan engellemez.

## 2. Veri ayrımı problemi

`data/test_new_knowledge.json` içinde **4** kayıt var. Bunların **2 tanesi eğitime birebir eş**. Dolayısıyla bağımsız değerlendirme yalnızca **2 yeni soruya** dayanıyor. Var olan tokenleştirici de eğitim+test metinlerinin karakterlerini sözlüğe dahil ediyor; bu deney daha temiz, train-only vocab ayrımının yerine geçmez.

## 3. Ölçülen erken durdurma testi

Sadece hesaplama farkı olan optimize AMN ile aynı repo Transformer'ı karşılaştırıldı.

- CPU, PyTorch 2.10.0+cpu, tek iş parçacığı; **3 seed (0/1/2)**.
- Öğrenme oranı **0,005**, batch **6**, d_model **32**, dizi uzunluğu **96**.
- En fazla **250 adım**, eğitim doğruluğunu **25 adımda bir** kontrol.
- **Yalnızca eğitim** setinin kuramsal tavanı (431/438) ardışık iki kontrol noktasında görülürse durdurma.
- Test cevapları yalnızca eğitim **bittikten sonra** kontrol edildi.
- Buradaki süreler yalnızca eğitim ve periyodik eğitim doğruluk kontrollerini kapsar; test sırasında greedy cevap üretimi dahil değildir.

| Seed | Optimize AMN adım / sn | Transformer adım / sn | İkisinin yeni iki cevaptaki tam doğruluğu |
|---:|---:|---:|---:|
| 0 | 125 / 3,204 | 225 / 10,550 | %0 |
| 1 | 125 / 3,066 | 225 / 10,612 | %0 |
| 2 | 125 / 4,802 | 250 / 12,044 | %0 |
| **Ortalama** | **125 / 3,691 sn** | **233,3 / 11,069 sn** | **%0** |

Bu *yalnızca eğitim verisinin teorik tavanına ulaşma maliyeti* karşılaştırmasıdır. Optimize AMN bu ölçütte yaklaşık **3,0 kat** hızlı. Transformer seed 2'de ilk kez 125. adımda tavana ulaştı fakat 150. adımda düşüp ancak 225–250 arasında tekrar kararlı duruma geldi.

Seed 0 ve 1 için ayrıca **altın cevap karakterlerini modele vermeden**, sabit cevap uzunluğu kadar greedy üretim yapıldı. Yeni iki soruda **her iki mimarinin de tam cevabı %0**. Seed 2 için greedy üretim yapılmadı; yeni iki soruda teacher-forced tam cevap eşleşmesi %0.

Bu deneyin **%100 başarıya ulaştığı iddia edilmemektedir**. Öğrenme/Genelleme engeli devam ediyor. %98,40 token üst sınırını %100 diye pazarlamak bilimsel olarak yanlış olur.

## 4. Korunan ve eklenenler

- Önceki model, eğitim motoru, ağırlıklar ve bütün geçmiş kıyaslar **değiştirilmedi**.
- <../experiments/accuracy_ceiling_audit.py> ile matematiksel veri denetimi eklendi.
- <../experiments/train_until_ceiling.py> ile yeniden üretilebilir, **test verisine bakmadan erken durdurma** deney scripti eklendi.
- <../tests/test_accuracy_ceiling_audit.py> ile `438 / 431 / 7` sınırı, teorik kayıp ve 2 kopya test kaydı regresyon testine bağlandı.
- <../results/comparison/early_stop_accuracy_ceiling_20261111.jsonl> içinde gerçek **6 deneylik** kısa ölçüm özeti saklanıyor.
- Yerel yürütme eski AMN ve Transformer denklemlerinin öz-yeterli PyTorch kopyası üzerinde yapıldı. Depoya eklenen yeniden üretim betiği gerçek `src` modüllerini kullanır; bu repo betiğinin tam eğitim çıktılarını önceki yerel ölçümlerle birebir eşleştirdiğimiz iddia edilmiyor.

## 5. Doğru sonraki kalite eşiği

**Kalite kabul şartı**: Geliştirme setinde daha önce görülmeyen kişi–özellik–değer bağlamlarından bağımsız tam cevap başarı oranı %100, ardından eğitim ve ayar sırasında hiç görülmemiş sabit test setinde %100. Sonuçlar **seed bazında**, yeterli sayıda örnekle ve **serbest üretim** üzerinden ölçülecek. Veri ile hedef cevap arasındaki ilişki açıkça tanımlanmalı; aynı girdiden iki farklı altın cevap beklenmemeli.

**Hız şartı**: Bu kalite eşiğini ilk sağlayan checkpoint'in toplam eğitim süresi; başarısız modellerin sırf daha hızlı eğitim tavanına ulaşması kalite zaferi olarak sayılmayacak.

Bu şartlar henüz karşılanmadığından **varsayılan model değiştirilmedi**. Önce bağımsız, kapsamlı, çelişkisiz Türkçe ilişkisel QA veri ayrımı ve cevap merkezli değerlendirme hazırlanmalıdır.

### Komutlar

```bash
python experiments/accuracy_ceiling_audit.py
python -m pytest -q tests/test_accuracy_ceiling_audit.py
python experiments/train_until_ceiling.py --models optimized transformer --seeds 0 1 2 --lr 0.005 --max-steps 250 --interval 25
```
