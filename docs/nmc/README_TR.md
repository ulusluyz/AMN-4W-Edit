# AMN-NMC: CPU üzerinde genel amaçlı nöral dil modeli (ilk deneysel sürüm)

Bu dal, AMN-4W-Edit projesinde önceki sürümleri değiştirmeden **ham metinden next-byte öğrenen**, Transformer kullanmayan, öğrenilebilir bellekli bir temel model araştırma çekirdeğidir. **AGI başarısı veya anlamlı Türkçe üretim henüz gösterilmemiştir.** Model hiçbir PDF/Word/Excel türüne, kişi/şehir şemasına, sabit ilişki türlerine veya hazır cevaplara bağlı değildir.

## Mimari

`UTF-8 byte 0..255 -> embedding -> 4 x NMCBlock -> 256 byte logits`.

Her blokta kapılı recurrent hidden-state, fast delta associative matrix, soft differentiable episodic slot bankası ve gated output mixer vardır. State: `hidden[B,d], fast[B,m,m], keys[B,S,m], values[B,S,m], usage[B,S], read[B,d]`.

Hızlı bellek: `F_new = lambda * F_old + eta * (v - F_old @ k) outer k`. `lambda` başlangıçta 0.99..1 aralığındadır. Slot belleği, softmax benzerlik ve kullanım üzerinden soft read, soft allocation ve soft write uygular. Bellek, sonraki byte cross-entropy kaybından gradient alır. Ayrık olay sınıflandırıcısı, parser, hard argmax adres veya hazır cevap tablosu kullanılmaz.

Nedensellik: token yalnızca önceki ve mevcut tokenın durumunu görür; okuma yazmadan önce gerçekleşir. Yeni durum sonraki tokenlara taşınır. CPU eğitiminde TBPTT kesmesi sadece chunk sınırında yapılır. Bellek boyutu sınırlıdır, interference/unutma/AGI çözülmüş değildir.

Kod yolu mevcut `src/amn4w.py` modülüyle çakışmamak için `src/amn_nmc/` olarak seçilmiştir.

## İlk model ve komutlar

`configs/nmc/cpu_initial.json` 256 byte sözlük, d=96, m=48, 32 slot, 4 blok ve **596.564 eğitilebilir parametre** içerir. CPU FP32, AdamW ve 64 byte TBPTT kullanır. Küçük `cpu_smoke.json` yalnız mekanizma kontrolü içindir.

```bash
python -m pip install torch pytest
PYTHONPATH=src python -m pytest -q tests/test_nmc.py
PYTHONPATH=src python -m amn_nmc.corpus --input /path/to/licensed_texts --output work/nmc_corpus --license VERIFIED_LICENSE
PYTHONPATH=src python -m amn_nmc.train --data work/nmc_corpus --output work/nmc_seed17 --config configs/nmc/cpu_initial.json --seed 17
PYTHONPATH=src python -m amn_nmc.generate --checkpoint work/nmc_seed17/last.pt --prompt 'Türkçe dil' --new-bytes 128
```

Bir `.txt` dosyası bir belge, `.jsonl` satırı bir belgedir (`{"text":"..."}`). Split, normalize metin SHA-256 ile **belge ayrık** train/validation/test oluşturur. Yakın kopya tespiti henüz yoktur. Kullanıcı yalnız lisansı doğrulanmış veri kullanmalıdır. Test kümesi ilk eğitim ve validation seçiminde okunmaz. Çıktı klasörü mevcutsa eğitim üzerine yazmaz. `initial.pt`, koşullu `best.pt`, `last.pt`, eğitim günlüğü ve ölçüm özeti oluşturur.

## İlk doğrulama ve sınırlar

Yerel CPU testleri: nedensellik, tam dizi/chunk eşitliği, hızlı/epizodik belleğe gradyan, bellek ablasyonu, TBPTT durum ayrıştırma, veri bölmesi, checkpoint ile üretim ve hatalı token kontrolü. İlk koşuda **9/9 geçti** (PyTorch 2.10.0+cpu). Bu sadece mekanizmayı doğrular, modelin dil yeteneğini veya genel zekâyı değil. Üretim rastgele başlangıçta anlamsız olabilir.

Sonraki kalite kapıları: gerçek lisanslı Türkçe korpusta üç seed ile held-out **bits-per-byte**; aynı byte/veri/işlem bütçeli GRU ve causal Transformer referansları; fast ve episodic ablation; uzun bağlam recall; bozulmadan bilgi güncelleme; OOD genelleme ve serbest Türkçe üretim testi. Bu sonuçlar elde edilene kadar büyük model/AGI iddiası yoktur.
