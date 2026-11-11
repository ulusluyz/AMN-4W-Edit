# Türkçe tam cümle: Modellerin gerçekten ürettiği metinler

Çıktılar yeni eğitilen ortak GRU çözücülerinin **kelime kelime ürettiği tahminlerdir**.
Cevap sırasında cümle birleştiren şablon kullanılmadı. Eğitim hedefleri sınırlı Türkçe şablonlarla oluşturuldu.
**Girdi serbest Türkçe değil, yapılandırılmış olgu üçlüleridir.**

| Model | Doğru tam cümle | Test | Öğrenilen decoder adımı |
|---|---:|---:|---:|
| dual_upsert (seed 0) | 1160 | 1200 | 400 |
| amn (seed 0) | 1153 | 1200 | 400 |
| transformer (seed 0) | 47 | 1200 | 650 |
| dual_upsert (seed 1) | 1200 | 1200 | 650 |
| amn (seed 1) | 1200 | 1200 | 650 |
| transformer (seed 1) | 2 | 1200 | 650 |
| dual_upsert (seed 2) | 1156 | 1200 | 350 |
| amn (seed 2) | 1156 | 1200 | 400 |
| transformer (seed 2) | 80 | 1200 | 650 |

## Örnek 1
Olgu: Can/konum/bahçe, Yaren/konum/park, Okan/konum/park, Elif/konum/kafe
Soru: Elif / konum
**Beklenen cümle:** Elif şu anda kafededir.
- **dual_upsert seed 0:** Elif şu anda kafededir. (doğru)
- **amn seed 0:** Elif şu anda kafededir. (doğru)
- **transformer seed 0:** Elif şu anda parktadır. (yanlış)
- **dual_upsert seed 1:** Yiğit'in kaleminin rengi sarıdır. (doğru)
- **amn seed 1:** Yiğit'in kaleminin rengi sarıdır. (doğru)
- **transformer seed 1:** Yiğit'in kaleminin rengi mavidir. (yanlış)
- **dual_upsert seed 2:** Ayşe'nin kaleminin rengi sarıdır. (doğru)
- **amn seed 2:** Ayşe'nin kaleminin rengi sarıdır. (doğru)
- **transformer seed 2:** Ayşe'nin kaleminin rengi sarıdır. (doğru)

## Örnek 2
Olgu: Yiğit/konum/salon, Nisa/renk/sarı, Yiğit/renk/sarı, Murat/renk/sarı
Soru: Yiğit / renk
**Beklenen cümle:** Yiğit'in kaleminin rengi sarıdır.
- **dual_upsert seed 0:** Yiğit'in kaleminin rengi sarıdır. (doğru)
- **amn seed 0:** Yiğit'in kaleminin rengi sarıdır. (doğru)
- **transformer seed 0:** Yiğit'in kaleminin rengi sarıdır. (doğru)
- **dual_upsert seed 1:** Zeynep şu anda garajdadır. (doğru)
- **amn seed 1:** Zeynep şu anda garajdadır. (doğru)
- **transformer seed 1:** Zeynep şu anda evdedir. (yanlış)
- **dual_upsert seed 2:** Zeynep şu anda garajdadır. (doğru)
- **amn seed 2:** Zeynep şu anda garajdadır. (doğru)
- **transformer seed 2:** Zeynep şu anda okuldadır. (yanlış)

## Örnek 3
Olgu: Umut/konum/garaj, Ece/konum/okul, Umut/renk/siyah, Zeynep/konum/kafe
Soru: Umut / konum
**Beklenen cümle:** Umut şu anda garajdadır.
- **dual_upsert seed 0:** Umut şu anda garajdadır. (doğru)
- **amn seed 0:** Umut şu anda garajdadır. (doğru)
- **transformer seed 0:** Umut şu anda okuldadır. (yanlış)
- **dual_upsert seed 1:** Yaren şu anda okuldadır. (doğru)
- **amn seed 1:** Yaren şu anda okuldadır. (doğru)
- **transformer seed 1:** Yaren şu anda bahçededir. (yanlış)
- **dual_upsert seed 2:** Murat'ın kaleminin rengi sarıdır. (doğru)
- **amn seed 2:** Murat'ın kaleminin rengi sarıdır. (doğru)
- **transformer seed 2:** Murat'ın kaleminin rengi yeşildir. (yanlış)

## Örnek 4
Olgu: Okan/konum/kafe, Nisa/renk/sarı, Deniz/konum/okul, Yaren/konum/park
Soru: Yaren / konum
**Beklenen cümle:** Yaren şu anda parktadır.
- **dual_upsert seed 0:** Yaren şu anda parktadır. (doğru)
- **amn seed 0:** Yaren şu anda parktadır. (doğru)
- **transformer seed 0:** Yaren şu anda kafededir. (yanlış)
- **dual_upsert seed 1:** Murat'ın kaleminin rengi sarıdır. (doğru)
- **amn seed 1:** Murat'ın kaleminin rengi sarıdır. (doğru)
- **transformer seed 1:** Murat'ın kaleminin rengi turuncudur. (yanlış)
- **dual_upsert seed 2:** Ece şu anda okuldadır. (doğru)
- **amn seed 2:** Ece şu anda okuldadır. (doğru)
- **transformer seed 2:** Ece şu anda garajdadır. (yanlış)

## Örnek 5
Olgu: Murat/renk/mor, Yiğit/konum/salon, Can/konum/bahçe, Ayşe/renk/sarı
Soru: Ayşe / renk
**Beklenen cümle:** Ayşe'nin kaleminin rengi sarıdır.
- **dual_upsert seed 0:** Ayşe'nin kaleminin rengi sarıdır. (doğru)
- **amn seed 0:** Ayşe'nin kaleminin rengi sarıdır. (doğru)
- **transformer seed 0:** Ayşe'nin kaleminin rengi mordur. (yanlış)
- **dual_upsert seed 1:** Mert şu anda kafededir. (doğru)
- **amn seed 1:** Mert şu anda kafededir. (doğru)
- **transformer seed 1:** Mert şu anda parktadır. (yanlış)
- **dual_upsert seed 2:** Can şu anda garajdadır. (yanlış)
- **amn seed 2:** Can şu anda garajdadır. (yanlış)
- **transformer seed 2:** Can şu anda okuldadır. (yanlış)

## Örnek 6
Olgu: Murat/renk/sarı, Sena/konum/salon, Yiğit/renk/turuncu, Sena/renk/sarı
Soru: Sena / renk
**Beklenen cümle:** Sena'nın kaleminin rengi sarıdır.
- **dual_upsert seed 0:** Sena'nın kaleminin rengi mordur. (yanlış)
- **amn seed 0:** Sena'nın kaleminin rengi mordur. (yanlış)
- **transformer seed 0:** Sena'nın kaleminin rengi beyazdır. (yanlış)
- **dual_upsert seed 1:** Yiğit şu anda salondadır. (doğru)
- **amn seed 1:** Yiğit şu anda salondadır. (doğru)
- **transformer seed 1:** Yiğit şu anda bahçededir. (yanlış)
- **dual_upsert seed 2:** Ayşe'nin kaleminin rengi sarıdır. (doğru)
- **amn seed 2:** Ayşe'nin kaleminin rengi sarıdır. (doğru)
- **transformer seed 2:** Ayşe'nin kaleminin rengi mavidir. (yanlış)

## Örnek 7
Olgu: Sena/renk/sarı, Can/konum/salon, Yaren/konum/okul, Deniz/konum/okul
Soru: Yaren / konum
**Beklenen cümle:** Yaren şu anda okuldadır.
- **dual_upsert seed 0:** Yaren şu anda okuldadır. (doğru)
- **amn seed 0:** Yaren şu anda okuldadır. (doğru)
- **transformer seed 0:** Yaren şu anda bahçededir. (yanlış)
- **dual_upsert seed 1:** Sena'nın kaleminin rengi sarıdır. (doğru)
- **amn seed 1:** Sena'nın kaleminin rengi sarıdır. (doğru)
- **transformer seed 1:** Sena'nın kaleminin rengi beyazdır. (yanlış)
- **dual_upsert seed 2:** Ece şu anda okuldadır. (doğru)
- **amn seed 2:** Ece şu anda okuldadır. (doğru)
- **transformer seed 2:** Ece şu anda kafededir. (yanlış)

## Örnek 8
Olgu: Zeynep/konum/kafe, Can/konum/bahçe, Deniz/renk/beyaz, Deniz/konum/okul
Soru: Zeynep / konum
**Beklenen cümle:** Zeynep şu anda kafededir.
- **dual_upsert seed 0:** Zeynep şu anda kafededir. (doğru)
- **amn seed 0:** Zeynep şu anda kafededir. (doğru)
- **transformer seed 0:** Zeynep şu anda okuldadır. (yanlış)
- **dual_upsert seed 1:** Sena şu anda salondadır. (doğru)
- **amn seed 1:** Sena şu anda salondadır. (doğru)
- **transformer seed 1:** Sena şu anda bahçededir. (yanlış)
- **dual_upsert seed 2:** Yaren şu anda parktadır. (doğru)
- **amn seed 2:** Yaren şu anda parktadır. (doğru)
- **transformer seed 2:** Yaren şu anda garajdadır. (yanlış)
