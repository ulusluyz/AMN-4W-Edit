# AMN-NMC — Tek onaylanan PDF kaynağına kilitli CPU eğitimi

## Aktif veri kararı

Bu dalda eski Wikipedia korpusu ve NMC'ye özgü Wikipedia örnekleme işi kaldırılmıştır. **Aktif AMN-NMC eğitim kaynağı yalnız kullanıcının yüklediği 80 sayfalık PDF dosyasıdır**. Tam PDF metni ve türetilen korpus herkese açık GitHub'a konulmaz.

İzin verilen kaynak PDF SHA-256:

`bc79c1ffdb3e78327496cff4d7e922d613d60275904a141bcf469e68cf5e1d20`

Kaynak: yapay zekâ destekli bilgilendirici metin yazma öğretimi üzerine tek bir akademik çalışma. **Tek kaynak olduğu için genel Türkçe dil modellemesine yeterli değildir.** Belgede katılımcı öğrencilere ilişkin bilgiler ve araştırma izin numaraları bulunmaktadır. Veri hazırlayıcısı izin kimliklerinin geçtiği satırları, tablonun düz yazı oluşturmayan satırlarını ve 78–80. sayfalardaki ek formları çıkarır; orijinal PDF değiştirilmez.

## Gerçek PDF üzerindeki yerel hazırlama ölçümü

- 80 PDF sayfası; kullanılabilir 77 sayfa.
- Bölmeler: sayfa 1–63 eğitim; 64–70 geliştirme; 71–77 son test.
- Filtrelenmiş toplam metin: 289.981 UTF-8 bayt.
- Eğitim: 232.249 bayt; geliştirme: 28.401 bayt; test: 29.331 bayt.
- 3 izin-numarası satırı, 12 tablo satırı, 2 tekil tablo hücresi, 27 eksik sayfa başlangıcı, 32 yarım sayfa bitişi çıkarıldı.
- OCR gerekmedi; metin katmanı okunabilir.
- Aynı çalışmadan sayfa düzeyi bölme **bağımsız belge/test ayrımı değildir**. Sızıntı riski devam eder.

Sözlüksel/dilbilgisel ve olgusal doğruluk insan tarafından tek tek doğrulanmamıştır. Eserin yayın lisansı belirtilmediği için ham metin kamusal repoya taşınmaz.

## Kullanım — yalnız bu PDF

```bash
python -m pip install torch pymupdf pytest
PYTHONPATH=src python -m amn_nmc.pdf_corpus --pdf "/path/to/yapay_zeka_egitim_temiz_korpus(2).pdf" --output work/nmc_pdf_corpus
PYTHONPATH=src python -m amn_nmc.train_pdf --data work/nmc_pdf_corpus --output work/nmc_pdf_smoke17 --config configs/nmc/cpu_smoke.json --seed 17
PYTHONPATH=src python -m amn_nmc.generate --checkpoint work/nmc_pdf_smoke17/last.pt --prompt "Türkçe" --new-bytes 128
```

`train_pdf` komutu manifestte kaynak SHA-256'sını ve üç bölme dosyasının SHA-256'sını doğrular. Önceki Wikipedia korpusu veya rastgele bir eğitim klasörü bu girişten geçemez. Bu kontrol, verinin dilsel yeterliliğini kanıtlamaz.

**Önemli:** PDF yalnızca eğitim verisidir; nöral AMN mimarisi PDF'ye özel değildir. Bu tek kaynakta başarı, genel yapay zekâ veya genel Türkçe yeterliliği değildir.

Eski Wikipedia metinleri özellik dalının güncel dosyalarından silinmiştir. Daha eski Git commit'leri ve `main` dalı GitHub'da durur; sıradan Git silme commit'i geçmişi geri dönük olarak yok etmez.
