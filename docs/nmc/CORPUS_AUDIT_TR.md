# AMN-NMC — Türkçe ön eğitim korpusu denetimi

**Durum: ONAYLANMADI.** Ne otomatik Türkçe etiketi ne de Wikipedia kaynağı, metnin dil modeli eğitimine hazır olduğunu kanıtlar. Bu çalışma hazır cevap ya da belge türü sınıflandırması değil, genel amaçlı otoregresif dil modeli için veri değerlendirmesidir.

## Gerçek kaynak dosyalarının incelenmesi

AMN-4W-Edit ana dalındaki Wikipedia kaynak metinlerinde 273 kayıt ve 273 benzersiz tam metin var; medyan uzunluk yaklaşık 165 kelime. Kaynak URL ve revizyonlar mevcut. İlk temel taramada HTML/Wiki etiketleri, açık bağlantı veya Kaynakça/Ayrıca bakınız başlığı bulunmadı. Bu sonuç dilbilgisi ve bilgi doğruluğunu göstermez. Veri kapsamı genel dil modeli için çok dar.

Soru-cevap bölmeleri: eğitim 746, doğrulama 63, test 54 kayıt. Eğitimde yalnız 255 benzersiz cevap var; 688 soruda tekrarlanan üç basit soru biçiminden biri kullanılıyor. Bu veriler genel dil modelinin ana ön eğitimine aktarılmayacak.

Dış aday: https://huggingface.co/datasets/kaan39/turkish-wikipedia-dataset-clean (122 bin civarı Wikipedia maddesi; CC BY-SA 4.0 lisans beyanı). Veri önizlemesinde düzleştirilmiş listeler, Kaynakça/Ayrıca bakınız gibi kalıntılar ve yabancı terimler var. Yayıncı eksik maddeler, tekrarlar ve biçimlendirme kalıntıları olabileceğini açıkça belirtiyor.

## Yeni kaynak denetim altyapısı

Kaynak: src/amn_nmc/data_audit.py

Doğrudan UTF-8 TXT, JSONL text veya extract alanlarını denetler. Yaptığı tanı: eksik/bozuk JSON ve UTF-8, Unicode kontrol karakteri, HTML/Wiki kalıntısı, bağlantı ve arayüz artıkları, başlık kalıntıları, e-posta/IBAN görünümü, çok kısa/uzun metin, aşırı rakam/sembol, seyrek cümle, tekrar eden satırlar, zayıf Türkçe sinyali, benzersizlik ve yaklaşık SimHash yakın-kopya kontrolü. Türkçe stopword/harf istatistikleri yalnız **heuristik**, güvenilir dil kimliği modeli değildir. SimHash benzerlik taraması tüm yakın kopyaları bulma garantisi vermez.

Çıktı: candidate_only.jsonl, rejected_or_review.jsonl, audit.json. Aday metinler otomatik temiz veya eğitime onaylı sayılmaz: candidate_is_approved_for_training = false. Kaynak, lisans ve orijinal metin özellikleri gözden geçirilir.

Kaynak: src/amn_nmc/wiki_sample_audit.py

Hugging Face satır API'sinden 0, 24000, 48000, 72000, 96000 ve 120000 ofsetlerinden gerçek kaynak metinleri kesmeden alır ve kalite raporu oluşturur. Bu sadece örneklem denetimidir, tam veri kümesi denetimi değildir. API erişilemezse doğrulama yapılmış sayılmaz; hiçbir eğitim başlatılmaz.

## Eğitim öncesi zorunlu açık kontroller

1. Tam korpusun her kaydını işleme ve kaynak kimliğini sabitleme.
2. Gerçek Türkçe dil tanılama + farklı dillerden geçerli bilimsel adlar ve alıntılar için manuel doğrulama.
3. Rastgele ve risk bazlı 100+ kaydı dilbilgisi, anlam, dil, zararlı içerik, doğruluk ve lisans açısından insan denetimi.
4. Kopya kümelerini train/validation/test arasında ayırma; benchmark kopyalarını dışlama.
5. Çok çeşitli konu ve yazı biçimlerine sahip, açıkça izlenebilir kaynak karması; telif atıf zinciri.
6. Temizlik öncesi/sonrası oran ve örneklerinin raporlanması; hiçbir bozuk metni sessizce düzeltmeme.

Eğitim veri onayı verilmeden geçerli held-out BPB veya anlam kalitesi ilan edilmeyecek. Eski checkpoint'ler değiştirilmeyecek.

## Kullanım

Python 3.11, CPU PyTorch (paket yükleme) ve pytest sonrasında:

    PYTHONPATH=src python -m pytest -q tests/test_nmc_data_audit.py
    PYTHONPATH=src python -m amn_nmc.data_audit --input data/amn_world_v1/accepted_pages.jsonl --output work/audit_existing --license CC-BY-SA-4.0
    PYTHONPATH=src python -m amn_nmc.data_audit --input data/amn_world_v1/train.jsonl --output work/audit_qa --license CC-BY-SA-4.0
    PYTHONPATH=src python -m amn_nmc.wiki_sample_audit --output work/wiki_sample

Lisans parametresi hak doğrulaması değil, yalnız kaynak beyanıdır.
