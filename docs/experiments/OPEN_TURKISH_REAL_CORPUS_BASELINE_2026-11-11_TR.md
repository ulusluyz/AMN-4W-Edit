# AMN-4W-Edit: Serbest Türkçe metinden metne ilk gerçek korpus sınaması (11.11.2026)

**Durum: Teknik eğitim başarılı, serbest Türkçe anlam kalitesi başarısız.** Sonuçlar model checkpoint'inden GitHub Actions üzerinde oluşturuldu. Bu deney eski AMN/Transformer/Dual checkpoint'lerini değiştirmez.

## Veri ve eğitim protokolü
- Kaynak: TFLai/Turkish-Alpaca, Apache-2.0: https://huggingface.co/datasets/TFLai/Turkish-Alpaca
- Kaynak: 51.914 yönerge-yanıt kaydı. Dizileme/uzunluk filtrelerinden sonra 14.180 uygun, 12.000 seçilmiş.
- Eğitim: 10.321; doğrulama: 861; test: 818. Normalleştirilmiş prompt SHA-256 ile ayrıldı.
- Kaynak dosya SHA-256: `f1aed098ae68849a66075220c7aae09971e0193371937c0f35433b2afd9d2f64`.
- İlk bağımsız model: 194.515 parametre, bayt tokenleştirici (her UTF-8 girdi desteklenir), 2 katmanlı GRU + nedensel ilişkisel AMN; Transformer yok.
- Eğitim: 1.200 AdamW güncelleme, 322,14 CPU saniyesi, cevap tokenlarına CE.

## Ölçüm
- En iyi doğrulama NLL: **1,5957 nat/bayt**.
- Görülmemiş test NLL: **1,6381 nat/bayt** (21.960 değerlendirilen cevap baytı).
- Test PPL: **5,1456** (bayt düzeyi; cümle doğruluğu değildir).
- Eğitim step 1 validation NLL 5,163 → step 1200 validation NLL 1,596.

### DÜZENLENMEMİŞ gerçek model çıktıları

Soru: Bir ağ bağlantısındaki gecikme neden artabilir?

Model: **Bir kütü bir küllük gibi.**

Soru: Bir kitabın ana fikri nasıl belirlenir?

Model: **Almanya konusu için maliyetin kalbol önerik.**

Soru: Yeni bir proje geliştirirken karşılaşılabilecek sorunları açıklar mısın?

Model: **Dünya'dır.**

Bunlar özgün, serbest model örneklemesidir; ancak **anlamsal, akıcı ve uzun Türkçe yanıt olarak yetersizdir**. Bu nedenle hedef **sağlanmadı**.

## Başlatılan ikinci mimari

`experiments/amn_open_turkish_sp.py`: GRU+AMN belleği korunurken, byte düzeyi yerine **train-only SentencePiece BPE alt sözcük** tokenleştirici kullanılır. Ek model checkpoint'i farklı bir dizinde saklanacaktır. Modelin teknik testleri yerelde geçti; eğitim/test sonucu henüz bu rapora eklenmedi. İkinci eğitim için iş akışı: `.github/workflows/amn_open_turkish_sp.yml`.

## İlgili gerçek çıktı dosyaları
- `results/open_turkish/metrics.json`
- `results/open_turkish/generated_turkish.md`
- `results/open_turkish/best.pt`
- `results/open_turkish/data_manifest.json`

**Bilimsel sınır:** SFT verisi ve bu ölçekte kısa eğitim, genel Türkçe anlama garantisi sağlamaz. Düşük test token kaybını yüksek cevap doğruluğu diye sunmak yanlış olur.