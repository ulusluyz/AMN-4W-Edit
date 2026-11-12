# AMN Dünya Bilgisi v1 — Kaynak ve Veri Kalite Denetimi

Denetim tarihi: 12 Kasım 2026.
Kaynaklar: Türkçe Wikipedia, TÜİK 2025 ADNKS bebek isimleri, MDN, DSÖ, BirdLife International.

## Kaynak ve ön temizlik
- Önceden belirlenmiş Türkçe Wikipedia maddesi: 355 (9 kategori).
- İndirilen: 350; indirilemeyen: 5.
- Otomatik incelemede ilk kabul: 315; ilk ret: 40.
- Kabul edilmiş ilk cümlelerin ayrıca gözle denetlenmesinde ret: 42.
- Son kabul edilen Wikipedia maddesi: 273; toplam ret: 82.
- Tek tek ret gerekçeleri data/amn_world_manual_review_exclusions_v1.json ve data/amn_world_v1/rejected_pages.jsonl içindedir.
- Kaynak URL, varsa revision ve ham metin SHA ayrıdır.

Ret örnekleri: Batarya (topçu birliği tanımı), Matris (anlam ayrımı), Afrika (yanlış nüfus yoğunluğu iddiası), Sıcaklık (döngüsel tanım), Donanım (anlam ayrımı), Diş (bozuk ifade), Virüs (yanıltıcı tanım), Yumurta (besin yerine biyolojik hücre anlamı).

## Temizlenmiş eğitim/kontrol/test
| Bölüm | Soru-cevap |
|---|---:|
| Eğitim | 746 |
| Doğrulama | 63 |
| Test | 54 |
| Toplam | 863 |

- 819 Wikipedia sorusu, 29 TÜİK isim sorusu, 7 DSÖ, 5 MDN, 3 BirdLife.
- 273 benzersiz Wikipedia kaynak URL'si ve toplam 282 kaynak konu ailesi.
- 294 farklı kaynak cevap metni ve 863 farklı soru.
- Tam soru tekrarı: 0; çelişen aynı soru cevapları: 0.
- Aynı Wikipedia URL'sinin train/val/test gruplarına bölünmesi: 0.
- Her Wikipedia kaynak gerçeği üç soru varyantıyla temsil ediliyor; toplam 863 satır 863 bağımsız bilgi demek değildir.
- Cevap uzunluğu 4–40 kelime, ortalama 16 kelime.
- Kaynak ham SHA-256: 249b0d07b564bc76f81a5897e1ece8b7641d8d4f2b74f62083f5da9d4ddab098.

## Atıf ve önemli sınırlar
Wikipedia metinleri CC BY-SA/GFDL atıf ve uygun kullanım koşulları gerektirir.
WHO, MDN ve BirdLife yanıtları kaynaklardan birebir alıntı değildir; her biri kaynak URL'siyle etiketlenmiş kısa editoryal parafrazdır.
Bu kaynak denetimi her olgunun bağımsız uzman tarafından yüzde yüz doğrulandığı anlamına gelmez.
Veri kümesi genel bir insanın dünya bilgisi ölçeğinde değil, 273 kaynak maddenin ilk cümlesine dayalı küçük bir başlangıç eğitimidir.
Modellerin gerçek başarısı için bağımsız konu bölmesindeki kayıp ve ham doğrudan yanıtlar birlikte ölçülmelidir.