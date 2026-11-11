# AMN-4W-Edit — Serbest Türkçe metinden alt sözcük düzeyinde üretim

Bu çalışma ilk bayt modelinin yerine geçmez; **ayrı bir deney** olarak korunur.

- Girdi, serbest UTF-8 Türkçe soru/metindir. İnsan veya ilişki alanları hazır verilmez.
- Tokenleştirici: **SentencePiece BPE** (eğitim kümesi üzerinde sıfırdan öğrenilir; `byte_fallback=True`). Girdi alanı kapalı 20 isim/2 özellik sözlüğüne bağlı değildir.
- Çözücü: 2 katmanlı GRU denetleyicisi + **Transformer içermeyen**, pozitif anahtarlarla normalize, nedensel kümülatif AMN bellek (öğrenilmiş sorgu, anahtar, değer).
- Sadece asistan yanıtına çapraz-entropi uygulanır; kullanıcı sorusu koşullandırıcı girdidir.
- Üretim, sıfırdan başlayan `<BOS>` ve prompt üzerinden **kendi tahmini alt sözcüklerle** yapılır. Cevap şablonu ve kişi/konum regex'i yoktur.
- Eğitim verisi, `TFLai/Turkish-Alpaca` (**Apache-2.0**) ile ilk bayt deneyindeki gibi prompt-hash train/val/test bölmelerine ayrılır; SentencePiece **yalnız train üzerinde** eğitilir.
- Model/test veri SHA, tokenizer SHA, eğitim adımları, kayıplar, `EOS` durumları ve üretilen ham Türkçe cevaplar arşivlenir.

## Komutlar

```bash
pip install torch sentencepiece numpy pytest
python experiments/prepare_open_turkish.py --output data/open_turkish --max-samples 12000 --max-seq-len 320 --min-answer-bytes 50 --min-train 1500
python -m pytest -q tests/test_amn_open_turkish_sp.py
python experiments/train_open_turkish_sp.py --data data/open_turkish --output results/open_turkish_sp --steps 1800 --max-len 192
python experiments/amn_open_turkish_sp.py --checkpoint results/open_turkish_sp/best.pt --tokenizer results/open_turkish_sp/tokenizer.model --prompt 'Türkçede bir kelime nasıl öğrenilir?' --max-new-tokens 160
```

**Güvenilirlik:** Eğitimden sonra metinler anlamsız veya tekrarlı çıkarsa **serbest Türkçe hedefi başarılı sayılmaz**. Token düzeyinde NLL/perplexity, farklı tokenizerların farklı token tanımları nedeniyle doğrudan kıyaslanamaz. Eski tüm ağırlıklar korunur.
