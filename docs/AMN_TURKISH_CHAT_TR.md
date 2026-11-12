# AMN-4W-Edit — Yerel Türkçe sohbet ve eğitim sonrası soru-cevap

## Gerçek durum

- Türkçe alt sözcük AMN modeli **2.200 adım eğitildi**, checkpoint GitHub'a kaydedildi.
- Model yolu: results/open_turkish_retrain_v2/best.pt
- Tokenizer: results/open_turkish_retrain_v2/tokenizer.model
- Parametre: **1.167.424**; eğitim CPU üzerinde **661 saniye**.
- Test NLL: **3,8621**; test perplexity: **47,56**.
- Dil kalitesi halen yetersiz. Cevaplar tekrarlı, anlamsız veya soruyla ilişkisiz olabilir. Eski bayt model ile PPL doğrudan kıyaslanamaz; tokenler farklı.

Bu sohbet kullanıcı metnini SentencePiece'e verir, cevabı kaydedilmiş AMNSubwordLM.generate yöntemiyle üretir. İnsan tarafından hazır cevap hazırlanmaz, ChatGPT yanıtı modele aktarılmaz. Harici çıkarım API'si kullanılmaz.

## Debian 13'te çalıştırma

Gereken: Python 3, python3-venv, ilk kurulumda pip paket indirme erişimi.

~~~bash
git clone https://github.com/ulusluyz/AMN-4W-Edit.git
cd AMN-4W-Edit
bash scripts/run_amn_turkish_chat.sh
~~~

Tarayıcıda: **http://127.0.0.1:8090**

Var olan depo kopyasında önce güncelleme:

~~~bash
git pull --ff-only
bash scripts/run_amn_turkish_chat.sh
~~~

Başlatıcı otomatik olarak .venv-amn-chat adlı bağımsız Python sanal ortamını kurar ve bağımlılıkları yalnız ilk sefer yükler. Eğitim başlatılmaz. Ctrl+C sunucuyu durdurur.

Alternatif komut:

~~~bash
python3 -m venv .venv-amn-chat
.venv-amn-chat/bin/python -m pip install -r requirements.txt
.venv-amn-chat/bin/python experiments/amn_chat_server.py --port 8090
~~~

- HTTP sunucusu varsayılan olarak 127.0.0.1'e bağlanır ve başka cihazlara açılmaz.
- Soru gönderme: POST /api/chat; örnek JSON: {"message":"Python for döngüsü nedir?","max_new_tokens":96,"temperature":0.7}
- Sağlık kontrolü: GET /api/health
- Sohbet ekranı mesaj geçmişini gösterir. İstenirse önceki iki turu modele dahil etme açılabilir, ancak model çoğunlukla tek turluk veriyle eğitildiğinden güvenilir çok-turlu bağlam yönetimi beklenmez.
- Cevaplar doğrudan modelin ham çıktılarıdır; otomatik düzeltme veya şablon kullanılmaz.

## Eğitimden sonra yeni sorular

experiments/qa_posttrain_v2.py, kayıtlı v2 checkpoint'ine robotik, Python, elektrik, kablosuz ağ, yazılım testi ve günlük Türkçe konularında altı soru sorar. Ek eğitim yoktur.

Çıktı konumları:
- results/open_turkish_retrain_v2/chat_probes/answers.json
- results/open_turkish_retrain_v2/chat_probes/answers_tr.md

.github/workflows/amn_turkish_chat_qa.yml otomatik olarak gerçek checkpoint'i yükleyerek API testi yapar ve altı ham cevabı arşivler.

## Proje bütünlüğü

Mevcut AMN-4W, Dual Upsert ve Transformer modelleriyle tüm eski checkpointler korunur. AMN-4W-jules salt okunur kalır.

**Durum: Etkileşimli test arayüzü var; henüz güvenilir bir genel Türkçe asistan değil.**
