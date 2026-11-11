# Ham Türkçe girişten modelin ürettiği uzun cevaplar

Girdi metinleri kural tabanlı kişi/ilişki/değer ayrıştırıcısına verilmez; öğrenilmiş byte-GRU kullanılır.
Çıkış kelimeleri GRU tarafından seçilir. **Eğitim hedefleri sınırlı elle yazılmış cümle tiplerinden oluşur.**
Bu bir genel Türkçe LLM veya sınırsız serbest metin üretimi değildir.

## Sonuçlar

| Test | Kabul | Tam doğru uzun cevap | Kapsam |
|---|---:|---:|---:|
| Bilinen ifade türleri | 600/600 | 593/600 | %98.83 |
| Eğitimde görülmemiş ifade kalıpları | 600/600 | 584/600 | %97.33 |

## Bilinen cümle tipleri

### 1. örnek
**Ham bilgi:** Elif şu anda kafede. Bulunduğu yeri sorarsan Zeynep kafede. Murat'ın kaleminden söz ediliyor ve renk sarı. Bulunduğu yeri sorarsan Yiğit salonda.
**Ham soru:** Zeynep'in bulunduğu yer hakkında bilgi verir misin?
**Gerçek model çıktısı:** Zeynep şu anda kafededir.
**Beklenen:** Zeynep şu anda kafededir.
**Durum:** DOĞRU

### 2. örnek
**Ham bilgi:** Mina için kalem rengi beyaz olarak belirtildi. Sena için son konum salonda kaydedildi. Elif şu anda kafede. Sena hakkında kaleminin sarı olduğu bilgisi var.
**Ham soru:** Sena nerede?
**Gerçek model çıktısı:** Verilen açıklamaya göre Sena şu anda salondadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer salonda konumudur.
**Beklenen:** Verilen açıklamaya göre Sena şu anda salondadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer salonda konumudur.
**Durum:** DOĞRU

### 3. örnek
**Ham bilgi:** Yiğit halen salonda görünüyor. Yaren şu anda parkta. Yiğit'in kalemi turuncu görünüyor. Murat hakkında kaleminin mavi olduğu bilgisi var.
**Ham soru:** Yiğit'in kalemi hangi renktedir?
**Gerçek model çıktısı:** Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Yiğit'in kaleminin rengi turuncudur. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Yiğit için belirtilen renk turuncu olarak görünüyor.
**Beklenen:** Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Yiğit'in kaleminin rengi turuncudur. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Yiğit için belirtilen renk turuncu olarak görünüyor.
**Durum:** DOĞRU

### 4. örnek
**Ham bilgi:** Yiğit'in kalemi turuncu renkte. Ece şu anda okulda. Mert'in konumu kafede olarak bildirildi. Konum kaydında Sena salonda olarak geçiyor.
**Ham soru:** Ece nerede bulunmuş?
**Gerçek model çıktısı:** Ece şu anda okuldadır.
**Beklenen:** Ece şu anda okuldadır.
**Durum:** DOĞRU

### 5. örnek
**Ham bilgi:** Sena'nın şu an bulunduğu mekân salonda. Kaleminin rengini sorarsan Sena için cevap sarı. Yiğit hakkında kaleminin sarı olduğu bilgisi var. Mina halen parkta görünüyor.
**Ham soru:** Kalemi hakkında Sena için hangi renk yazılmış?
**Gerçek model çıktısı:** Metindeki bilgiye göre Sena'nın kaleminin rengi sarıdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap sarı rengidir.
**Beklenen:** Metindeki bilgiye göre Sena'nın kaleminin rengi sarıdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap sarı rengidir.
**Durum:** DOĞRU

### 6. örnek
**Ham bilgi:** Yiğit'in konumu salonda olarak bildirildi. Bulunduğu yeri sorarsan Ece okulda. Selin için yer bilgisi garajda. Deniz şu anda okulda.
**Ham soru:** Selin için hangi konum bildirilmiş?
**Gerçek model çıktısı:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Selin şu anda garajdadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Selin için kayıtlı yer garajda olarak belirtilmiştir.
**Beklenen:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Selin şu anda garajdadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Selin için kayıtlı yer garajda olarak belirtilmiştir.
**Durum:** DOĞRU

### 7. örnek
**Ham bilgi:** Kaleminin rengini sorarsan Deniz için cevap beyaz. Yaren için yer bilgisi parkta. Murat'ın kaleminden söz ediliyor ve renk sarı. Konum kaydında Okan kafede olarak geçiyor.
**Ham soru:** Yaren şu anda nerede bulunuyor?
**Gerçek model çıktısı:** Yaren şu anda parktadır.
**Beklenen:** Yaren şu anda parktadır.
**Durum:** DOĞRU

### 8. örnek
**Ham bilgi:** Murat'ın kaleminden söz ediliyor ve renk mor. Bulunduğu yeri sorarsan Zeynep garajda. Sena için son konum salonda kaydedildi. Okan'ın şu an bulunduğu mekân parkta.
**Ham soru:** Zeynep adlı kişinin bulunduğu yeri açıklar mısın?
**Gerçek model çıktısı:** Verilen açıklamaya göre Zeynep şu anda garajdadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer garajda konumudur.
**Beklenen:** Verilen açıklamaya göre Zeynep şu anda garajdadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer garajda konumudur.
**Durum:** DOĞRU

### 9. örnek
**Ham bilgi:** Kaleminin rengini sorarsan Mina için cevap beyaz. Deniz'in kalemi beyaz renkte. Deniz'in şu an bulunduğu mekân okulda. Ece'nin şu an bulunduğu mekân okulda.
**Ham soru:** Deniz adlı kişinin bulunduğu yeri açıklar mısın?
**Gerçek model çıktısı:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Deniz şu anda okuldadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Deniz için kayıtlı yer okulda olarak belirtilmiştir.
**Beklenen:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Deniz şu anda okuldadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Deniz için kayıtlı yer okulda olarak belirtilmiştir.
**Durum:** DOĞRU

### 10. örnek
**Ham bilgi:** Sena hakkında kaleminin sarı olduğu bilgisi var. Umut için kalem rengi siyah olarak belirtildi. Selin şimdi parkta vakit geçiriyor. Mina halen parkta görünüyor.
**Ham soru:** Selin bugün hangi yerde?
**Gerçek model çıktısı:** Selin şu anda parktadır.
**Beklenen:** Selin şu anda parktadır.
**Durum:** DOĞRU

### 11. örnek
**Ham bilgi:** Bulunduğu yeri sorarsan Sena salonda. Sena'nın kaleminin rengi sarı. Renk bilgisi Umut için siyah olarak kaydedilmiş. Nisa'nın kaleminden söz ediliyor ve renk sarı.
**Ham soru:** Sena adlı kişinin kaleminin rengini açıklar mısın?
**Gerçek model çıktısı:** Metindeki bilgiye göre Sena'nın kaleminin rengi sarıdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap sarı rengidir.
**Beklenen:** Metindeki bilgiye göre Sena'nın kaleminin rengi sarıdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap sarı rengidir.
**Durum:** DOĞRU

### 12. örnek
**Ham bilgi:** Okan için yer bilgisi kafede. Ece için son konum okulda kaydedildi. Konum kaydında Elif kafede olarak geçiyor. Konum kaydında Yaren parkta olarak geçiyor.
**Ham soru:** Yaren şu anda nerede bulunuyor?
**Gerçek model çıktısı:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Yaren şu anda parktadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Yaren için kayıtlı yer parkta olarak belirtilmiştir.
**Beklenen:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Yaren şu anda parktadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Yaren için kayıtlı yer parkta olarak belirtilmiştir.
**Durum:** DOĞRU


## Yeni cümle kalıpları

### 1. örnek
**Ham bilgi:** Ece okulda olduğu bilgisi verildi. Zeynep, garajda zaman geçiriyor. Okan, kafede zaman geçiriyor. Yaren parkta olduğu bilgisi verildi.
**Ham soru:** Ece son olarak nerede görüldü?
**Gerçek model çıktısı:** Ece şu anda okuldadır.
**Beklenen:** Ece şu anda okuldadır.
**Durum:** DOĞRU

### 2. örnek
**Ham bilgi:** Yaren, parkta zaman geçiriyor. Deniz'in kullandığı kalem beyaz renkle tarif ediliyor. Okan, kafede zaman geçiriyor. Deniz, okulda zaman geçiriyor.
**Ham soru:** Deniz son olarak nerede görüldü?
**Gerçek model çıktısı:** Verilen açıklamaya göre Deniz şu anda okuldadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer okulda konumudur.
**Beklenen:** Verilen açıklamaya göre Deniz şu anda okuldadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer okulda konumudur.
**Durum:** DOĞRU

### 3. örnek
**Ham bilgi:** Deniz, beyaz renginde bir kaleme sahip. Deniz, okulda zaman geçiriyor. Murat, sarı renginde bir kaleme sahip. Selin, garajda zaman geçiriyor.
**Ham soru:** Deniz hangi renk ile ilişkilendiriliyor?
**Gerçek model çıktısı:** Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Deniz'in kaleminin rengi beyazdır. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Deniz için belirtilen renk beyaz olarak görünüyor.
**Beklenen:** Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Deniz'in kaleminin rengi beyazdır. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Deniz için belirtilen renk beyaz olarak görünüyor.
**Durum:** DOĞRU

### 4. örnek
**Ham bilgi:** Yiğit, turuncu renginde bir kaleme sahip. Nisa, sarı renginde bir kaleme sahip. Sena'nın kullandığı kalem yeşil renkle tarif ediliyor. Zeynep kafede olduğu bilgisi verildi.
**Ham soru:** Nisa hangi renk ile ilişkilendiriliyor?
**Gerçek model çıktısı:** Nisa'nın kaleminin rengi sarıdır.
**Beklenen:** Nisa'nın kaleminin rengi sarıdır.
**Durum:** DOĞRU

### 5. örnek
**Ham bilgi:** Yiğit, turuncu renginde bir kaleme sahip. Deniz, okulda zaman geçiriyor. Selin, parkta zaman geçiriyor. Deniz'in kullandığı kalem beyaz renkle tarif ediliyor.
**Ham soru:** Deniz son olarak nerede görüldü?
**Gerçek model çıktısı:** Verilen açıklamaya göre Deniz şu anda okuldadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer okulda konumudur.
**Beklenen:** Verilen açıklamaya göre Deniz şu anda okuldadır. Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer okulda konumudur.
**Durum:** DOĞRU

### 6. örnek
**Ham bilgi:** Zeynep, garajda zaman geçiriyor. Yaren, okulda zaman geçiriyor. Selin parkta olduğu bilgisi verildi. Murat'ın kullandığı kalem sarı renkle tarif ediliyor.
**Ham soru:** Yaren son olarak nerede görüldü?
**Gerçek model çıktısı:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Yaren şu anda okuldadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Yaren için kayıtlı yer okulda olarak belirtilmiştir.
**Beklenen:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Yaren şu anda okuldadır. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Yaren için kayıtlı yer okulda olarak belirtilmiştir.
**Durum:** DOĞRU

### 7. örnek
**Ham bilgi:** Okan parkta olduğu bilgisi verildi. Ece okulda olduğu bilgisi verildi. Yaren parkta olduğu bilgisi verildi. Nisa'nın kullandığı kalem sarı renkle tarif ediliyor.
**Ham soru:** Okan son olarak nerede görüldü?
**Gerçek model çıktısı:** Okan şu anda parktadır.
**Beklenen:** Okan şu anda parktadır.
**Durum:** DOĞRU

### 8. örnek
**Ham bilgi:** Ayşe, sarı renginde bir kaleme sahip. Sena, yeşil renginde bir kaleme sahip. Selin garajda olduğu bilgisi verildi. Nisa, sarı renginde bir kaleme sahip.
**Ham soru:** Ayşe'nin kullandığı kalemi tarif eder misin?
**Gerçek model çıktısı:** Metindeki bilgiye göre Ayşe'nin kaleminin rengi sarıdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap sarı rengidir.
**Beklenen:** Metindeki bilgiye göre Ayşe'nin kaleminin rengi sarıdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap sarı rengidir.
**Durum:** DOĞRU

### 9. örnek
**Ham bilgi:** Elif kafede olduğu bilgisi verildi. Murat'ın kullandığı kalem mor renkle tarif ediliyor. Okan kafede olduğu bilgisi verildi. Mert kafede olduğu bilgisi verildi.
**Ham soru:** Okan son olarak nerede görüldü?
**Gerçek model çıktısı:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Okan şu anda kafededir. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Okan için kayıtlı yer kafede olarak belirtilmiştir.
**Beklenen:** Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. Okan şu anda kafededir. Diğer kişilerin konumları farklı olabilir. Sonuç olarak Okan için kayıtlı yer kafede olarak belirtilmiştir.
**Durum:** DOĞRU

### 10. örnek
**Ham bilgi:** Mina, beyaz renginde bir kaleme sahip. Mina, parkta zaman geçiriyor. Murat, kırmızı renginde bir kaleme sahip. Umut'un kullandığı kalem siyah renkle tarif ediliyor.
**Ham soru:** Mina hangi renk ile ilişkilendiriliyor?
**Gerçek model çıktısı:** Mina'nın kaleminin rengi beyazdır.
**Beklenen:** Mina'nın kaleminin rengi beyazdır.
**Durum:** DOĞRU

### 11. örnek
**Ham bilgi:** Deniz, beyaz renginde bir kaleme sahip. Deniz okulda olduğu bilgisi verildi. Okan, kafede zaman geçiriyor. Murat, sarı renginde bir kaleme sahip.
**Ham soru:** Deniz hangi renk ile ilişkilendiriliyor?
**Gerçek model çıktısı:** Metindeki bilgiye göre Deniz'in kaleminin rengi beyazdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap beyaz rengidir.
**Beklenen:** Metindeki bilgiye göre Deniz'in kaleminin rengi beyazdır. Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap beyaz rengidir.
**Durum:** DOĞRU

### 12. örnek
**Ham bilgi:** Can salonda olduğu bilgisi verildi. Mina'nın kullandığı kalem beyaz renkle tarif ediliyor. Murat'ın kullandığı kalem mavi renkle tarif ediliyor. Yiğit'in kullandığı kalem sarı renkle tarif ediliyor.
**Ham soru:** Murat hangi renk ile ilişkilendiriliyor?
**Gerçek model çıktısı:** Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Murat'ın kaleminin rengi mavidir. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Murat için belirtilen renk mavi olarak görünüyor.
**Beklenen:** Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. Murat'ın kaleminin rengi mavidir. Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. Sonuç olarak Murat için belirtilen renk mavi olarak görünüyor.
**Durum:** DOĞRU


## Alan dışı sorular

**Soru:** Bir kara deliğin nasıl oluştuğunu anlatır mısın?
**Durum:** rejected
**Yanıt:** Unknown person / question

**Soru:** Bora hangi marka telefonu aldı?
**Durum:** rejected
**Yanıt:** Unknown person / question

**Soru:** Ali neden sinirli?
**Durum:** rejected
**Yanıt:** Unsupported question type

**Soru:** Yağmurun oluşumunu ayrıntılı açıklar mısın?
**Durum:** rejected
**Yanıt:** Unknown person / question
