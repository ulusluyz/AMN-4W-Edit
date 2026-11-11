# AMN-4W ile Transformer — 1.000 adımlık CPU karşılaştırması

## Kapsam

Karşılaştırma, mevcut AMN-4W v0.4 üretim modeli yerine **bağımsız sentetik AMN proxy** ile iki katmanlı Transformer arasında yapılmıştır. Görev: iki bağımsız A→B→C zincirinden, verilen A anahtarı için C'yi tahmin etmek.

- PyTorch: 2.10.0+cpu, tek CPU thread.
- 1.000 eğitim adımı, batch 64, V=50, zincir sayısı M=2, Adam LR=0,002.
- 5 seed (0–4), her seed için aynı modelden bağımsız oluşturulan eğitim akışı ve 500 ortak değerlendirme örneği.
- AMN: 31.988 parametre; Transformer: 73.842 parametre. **Parametre sayıları eşit değildir.**
- AMN proxy'ye ilişkiyi nereye yazacağını belirleyen `[0,1,1]` maskesi verilmiştir. Transformer bu maskeyi almaz. Yardımcı kayıplı AMN ayrıca B ara hedefiyle eğitilmiştir; bu nedenle bilgi/denetim eşitliği yoktur.
- İlgisiz veya hedef zinciri kaldırma değerlendirmesi eğitim dağılımı dışındadır (tokenlar 0 ile değiştirilir). Bunlar doğrudan bellek sızıntısı olarak yorumlanamaz.

## Sonuçlar (5 seed)

| Model | Nihai doğruluk ort ± SS | Süre ort ± SS (sn/seed) | İlgisiz zincir kaldırılınca doğruluk | Hedef zincir kaldırılınca doğruluk | İlgisiz zincir silindiğinde doğru cevabın olasılık değişimi |
|---|---:|---:|---:|---:|---:|
| AMN-4W proxy | %99,88 ± 0,18 puan | 11,37 ± 0,76 | %100 | %1,00 | 0,00502 |
| AMN-4W proxy + yardımcı kayıp | %100,00 ± 0 | 11,79 ± 0,88 | %100 | %0,56 | 0,00087 |
| Transformer 2 katman | %99,12 ± 0,78 puan | 6,73 ± 0,40 | %100 | %0,20 | 0,01310 |

Tekil nihai doğruluklar (seed 0–4):

| Mimari | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| AMN-4W proxy | 100,0 | 99,8 | 100,0 | 99,6 | 100,0 |
| AMN-4W proxy + yardımcı kayıp | 100,0 | 100,0 | 100,0 | 100,0 | 100,0 |
| Transformer | 99,6 | 99,6 | 99,8 | 98,6 | 98,0 |

## Yorum ve sınırlar

Bu deneyde AMN proxy yüksek doğruluğa ulaşırken Transformer, 350 adımda görülen farkı büyük oranda kapatmıştır. Transformer'ın eğitim süresi bu donanım ve küçük problemde daha kısa olmuştur. Bunlar **genel AMN-4W üstünlüğü**, geniş ölçekli hesaplama avantajı, gerçek dil performansı veya Transformer'da sızıntının nicel olarak AMN'den fazla olduğu anlamına gelmez. Eğitim bütçesi adım bazında eşit olsa da parametre, denetim ve yazma maskesi eşit değildir.

## Yeniden üretme

```bash
cd /mnt/data
python compare_amn_transformer.py --steps 1000 --seeds 5 --vocab 50 --chains 2 --batch 64 --eval 500 --out yeniden_1000.jsonl
```

Orijinal deneme donanım sınırı nedeniyle model/seed bazında ayrı süreçlerde tamamlanmıştır. Her koşunun sonucu `amn_transformer_results_1000_complete.jsonl` dosyasına yazılmıştır. Test kodu: `compare_amn_transformer.py`; AMN proxy bağımlılığı: `amn4w_local_test.py`.