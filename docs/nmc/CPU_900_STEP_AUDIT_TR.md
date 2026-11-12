# AMN-NMC — CPU nöral dil modeli pilot eğitim doğrulaması (2026-11-12)

## Hedef ve sınır

AMN-NMC'nin hedefi PDF'ye, Word'e, Excel'e veya sabit soru-cevap biçimlerine bağlı olmayan, ham byte dizilerinden otoregresif şekilde öğrenen özgün bir nöral dil modeli geliştirmektir. PDF yalnızca kullanıcının seçtiği veri kaynağıdır; model mimarisinde PDF ayrıştırıcısı, kişi–şehir şeması veya hazır cevap sistemi yoktur.

**Bu pilot AGI, genel Türkçe dil becerisi veya güçlü uzun süreli akıl yürütme kanıtı değildir.** Tek akademik çalışma, tek seed ve kısmi korpus geçişiyle sınanmıştır.

## Kaynak ve deney protokolü

- Kullanıcı PDF'si SHA-256: `bc79c1ffdb3e78327496cff4d7e922d613d60275904a141bcf469e68cf5e1d20`.
- Eğitim: sayfa 1–63, 232.249 UTF-8 byte; doğrulama: sayfa 64–70, 28.401 byte; test: sayfa 71–77, 29.331 byte.
- Tek kaynak olduğundan bölmeler **bağımsız belge testi değildir**.
- 900 gradient güncellemesi; 57.232 hedef byte; 16/63 eğitim sayfası görüldü; TBPTT 64 byte; CPU FP32; seed 17.
- AdamW, öğrenme oranı 0,0007; her 225 adımda ilk 4.096 doğrulama byte'ında ara kontrol; 900 adım sonrası tam doğrulama. Test, seçilmiş en iyi checkpoint üzerinde ayrı ölçüldü.
- AMN-NMC: 93.642 parametre, 2 blok, 48 gizli boyut, 24 ilişkisel bellek boyutu, 8 epizodik slot.
- Plain GRU: 91.376 parametre, 2 katman, gizli boyut 68. Aynı eğitim byte'ları, aynı sıra, aynı güncelleme sayısı ve optimizer. **GRU, AMN çekirdeğinin parçası değil**; yalnızca kontrol modelidir. İşlem maliyetleri ve gerçek CPU süreleri eşitlenmemiştir.

## Ölçülen değerler

| Metrik | AMN-NMC | GRU kontrol |
|---|---:|---:|
| Parametre | 93.642 | 91.376 |
| Eğitim adımı | 900 | 900 |
| Eğitim byte | 57.232 | 57.232 |
| Başlangıç doğrulama, ilk 4.096 byte (BPB) | 8,199387 | 8,001955 |
| Son doğrulama, tüm 28.394 hedef byte (BPB) | **2,924626** | 3,053676 |
| Ayrılmış test, tüm 29.324 hedef byte (BPB) | **2,870472** | 2,992248 |

Bir seed'deki küçük fark, genellenebilir mimari üstünlük kanıtı değildir. Başlangıç ve son doğrulama farklı kapsamda olduğundan doğrudan yüzde düşüş çıkarılmamalıdır. Test tek bir makalenin devam sayfalarıdır ve bağımsız dünya bilgisi testi değildir.

## Bellek için değerlendirme

900 adımlık tam modelin ağırlıkları korunarak yalnız **çıkarım sırasında** bellek alt bileşenleri kapatıldı. İlk 4.096 doğrulama byte'ında:

| Fast bellek | Epizodik bellek | BPB |
|---|---|---:|
| Açık | Açık | 3,221717 |
| Kapalı | Açık | 3,252711 |
| Açık | Kapalı | 3,239348 |
| Kapalı | Kapalı | 3,350249 |

Bu, modelin bellek okumalarını kullandığını gösterir; **ayrı eğitilmiş mimari ablasyonu değildir**. Her bellek varyantı için yeniden eğitim ve üç seed gerekir.

## Üretim testi

Checkpoint'ten şablonsuz doğrudan byte örnekleme yapıldı. Türkçe benzeri sözcük parçaları oluşsa da **üretilen metin anlaşılır, tutarlı Türkçe değildir**; bazen geçersiz UTF-8 byte dizileri oluşur. Hedef başarı kriteri karşılanmadı.

## Yeni kaynak kodu ve iyileştirmeler

- `src/amn_nmc/train.py`: gerçek TBPTT, tüm model+optimizer+fast/episodic runtime state kaydı, tam konumlu deterministik resume, belge SHA-256 değişim koruması, ara ve son doğrulama, atomik checkpoint, train byte sayaçları.
- `src/amn_nmc/train_pdf.py`: yalnız kullanıcı PDF'sinin sabit kimliğini ve üç split hash'ini kabul eden giriş; `--resume --extra-steps N` desteği.
- `src/amn_nmc/baseline_gru.py`: parametre sayısı yakın GRU kontrolü.
- `src/amn_nmc/evaluate_checkpoint.py`: son testin checkpoint'e bağlı bir kez kaydedilmesi.
- `tests/test_nmc_training_v2.py`: kesintisiz-resume ağırlık eşitliği, corpus değişim kilidi, BPB kapsamı, state Tensor kalıcılığı, uzun bağlam finite, bellek gradyanları ve parametre bandı kontrolü.

Kullanım (PDF ham metni veya korpusu GitHub'a yüklemeyin):

```bash
PYTHONPATH=src python -m amn_nmc.pdf_corpus --pdf '/path/to/yapay_zeka_egitim_temiz_korpus(2).pdf' --output work/nmc_pdf
PYTHONPATH=src python -m amn_nmc.train_pdf --data work/nmc_pdf --config configs/nmc/cpu_pilot_96k.json --output work/run17 --seed 17
PYTHONPATH=src python -m amn_nmc.train_pdf --data work/nmc_pdf --config configs/nmc/cpu_pilot_96k.json --output work/run17 --seed 17 --resume --extra-steps 300
PYTHONPATH=src python -m amn_nmc.baseline_gru --data work/nmc_pdf --config configs/nmc/cpu_pilot_96k.json --output work/gru17 --seed 17
PYTHONPATH=src python -m pytest -q tests/test_nmc.py tests/test_nmc_pdf_source.py tests/test_nmc_training_v2.py
```

## Açık konular

Modelde oturumlar arası semantik bellek pekiştirme, uzun vadeli nöral bilgi konsolidasyonu, geniş veri üzerinde güçlü dil üretimi, öğrenilmemiş görev genellemesi ve akıl yürütme halen doğrulanmış değildir. CPU'da üç seed, eğitimli bellek ablasyonları ve birbirinden bağımsız genel metin kaynakları üzerinde deney yapılmadan mimariyi AGI olarak nitelemek doğru değildir. Kullanıcının talimatıyla eski Wikipedia verileri **bu geliştirme dalındaki aktif eğitimden çıkarılmıştır**; ana daldaki eski Git geçmişi silinmemiştir.
