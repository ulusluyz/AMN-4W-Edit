# AMN-4W-Edit — Doğrudan Türkçe metinden metne üretim deneyi

## Hedef ve mimari

Bu deney, daha önceki **20 isim ve 2 özellik** içeren yapılandırılmış QA sisteminin yerine geçmez; tamamlayıcı, bağımsız bir **serbest metinden metne AMN** prototipidir.

- Türkçe kullanıcı girdisi herhangi bir UTF-8 metin olabilir. Girişte kişi/konum/renk kuralları, etiketler veya regex ayrıştırması yoktur.
- **Tek tek 259 bayt/bitiş/pad tokenı** işlenir; bilinmeyen sözcük yoktur.
- Yanıt, `<BOS>` ve kullanıcı metni sonrasından başlayarak **otoregresif bayt tahmini** ile üretilir; hiçbir hazır cevap şablonu birleştirilmez.
- Ağ: 2 katmanlı GRU denetleyicisi + **doğrusal zamanlı ilişkisel AMN bellek**, `S_t = S_(t-1) + k_t v_tᵀ`, öğrenilebilir sorgu ve normalize okuma. **Transformer yoktur.**
- Eğitim sırasında prompt tokenlarına loss uygulanmaz. Sadece gerçek yanıt baytları hedeflenir.
- Aynı ağırlıklar hem `forward` ile toplu eğitimde hem `step` ile akış halinde üretimde çalışır; toleranslı eşdeğerlik test edilir.
- Model **sıfırdan eğitilir**. OpenAI, ChatGPT veya başka bir önceden eğitilmiş modelin ağırlıklarını kullanmaz.

## Gerçek Türkçe eğitim verisi

Kaynak: [TFLai/Turkish-Alpaca](https://huggingface.co/datasets/TFLai/Turkish-Alpaca), **Apache-2.0**. Kaynak dosyanın SHA-256 özeti ve kullanılabilen örnek sayıları `data_manifest.json` dosyasına yazılır. Ham veri depoya gönderilmez. Bu bir instruction-response veri kümesidir, genel amaçlı pretraining korpusu değildir. Bazı cevapları yanlış veya anlamsız olabilir.

Aynı istemin farklı bölmelere sızmasını engellemek için normalleştirilmiş istem SHA-256 ile gruplanır, kopyalar elenir ve deterministik train/validation/test bölmeleri ayrılır. Model seçimi yalnız validation loss ile yapılır, test ilk kez model seçiminden sonra okunur. Bu ayrım yakın anlamlı kopyaların tümünü engellemez; kaynak içerik denetimi hâlâ gereklidir.

## Çalıştırma

```bash
python experiments/prepare_open_turkish.py --output data/open_turkish --max-samples 12000 --max-seq-len 320 --min-answer-bytes 50 --min-train 1500
python -m pytest -q tests/test_amn_open_turkish.py
python experiments/train_open_turkish.py --data data/open_turkish --output results/open_turkish --steps 1200 --max-len 320
python experiments/amn_open_turkish.py ask --checkpoint results/open_turkish/best.pt --prompt "Sence insanlar yeni bir beceriyi nasıl öğrenir?" --temperature 0.7 --top-k 12 --max-new-bytes 320
```

Çalıştırma tümüyle yereldir; yalnız **ilk açık veri indirmesi internet** gerektirir. Eğitim ve çıkarımda bulut API çağrısı yapılmaz.

## Kabul kriterleri

Sadece kodun çalışması **Türkçe anlama kanıtı değildir**. `metrics.json` dosyasındaki:

1. Model ağırlık sayısı, eğitim adımı, veri kümesi SHA-256 ve test ayrımı;
2. Görülmemiş test yanıtlarında token başına negatif log-likelihood (NLL) ve perplexity;
3. Eğitimde görülmeyen gerçek Türkçe istemlere verilen **ham, değiştirilmemiş model yanıtları**;
4. UTF-8 geçerliliği ve `EOS` ile durma oranları

birlikte değerlendirilmelidir. Basit modelin gerçek anlamlı, uzun, özgün Türkçe konuşabildiği **gösterilmedikçe bu hedef tamamlandı sayılmaz**. Mevcut AMN-4W-jules ve önceki checkpointler değiştirilmeyecek.
