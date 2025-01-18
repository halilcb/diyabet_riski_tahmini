# Diyabet Riski Tahmin Projesi

Bu proje, bireylerde diyabet olasılığını tahmin etmek için farklı makine öğrenimi modellerini kullanmaktadır.
Proje, veri ön işleme, keşfedici veri analizi, aykırı değer tespiti, model eğitimi, hiperparametre optimizasyonu ve değerlendirme aşamalarından oluşmaktadır.



## Veri Seti

Bu projede Pima Indians Diabetes Database kullanılmıştır. Veri seti, 768 bireyin tıbbi kayıtlarını içermekte ve her kayıtta 8 özellik ile diyabet durumunu belirten bir hedef değişken bulunmaktadır.

Özellikler şunlardır:

- **Pregnancies**: Gebelik sayısı
- **Glucose**: Plazma glikoz konsantrasyonu
- **BloodPressure**: Diyastolik kan basıncı (mm Hg)
- **SkinThickness**: Triseps deri kalınlığı (mm)
- **Insulin**: 2 saatlik serum insülin seviyesi (mu U/ml)
- **BMI**: Vücut kitle indeksi (kg/m^2)
- **DiabetesPedigreeFunction**: Diyabet soy ağacı fonksiyonu (aile hikayesine dayalı diyabet olasılığı)
- **Age**: Yaş (yıl)

Hedef değişken **Outcome** şu değerlere sahiptir:

- `0`: Diyabetik değil
- `1`: Diyabetik

Veri seti `diabetes.csv` dosyasında yer almaktadır.


## Proje Aşamaları

### 1. Veri Keşfi ve Görselleştirme

- Veri seti yüklenerek `info()` ve `describe()` ile genel yapısı incelendi.
- Özellikler ve hedef değişken arasındaki ilişkiler çiftli grafiklerle (örneğin, `pairplot`) görselleştirildi.
- Özellikler arasındaki korelasyon bir korelasyon haritası (örneğin, `heatmap`) ile analiz edildi.

### 2. Aykırı Değer Tespiti ve Temizleme

- Her sayısal kolon için **Interquartile Range (IQR)** yöntemi kullanılarak aykırı değerler tespit edildi.
- Aykırı değerler temizlenerek temizlenmiş bir veri seti oluşturuldu.

### 3. Veri Ayrımı ve Standardizasyon

- Veri, `train_test_split` ile eğitim (%75) ve test (%25) olmak üzere ikiye ayrıldı.
- Veriler, `StandardScaler` kullanılarak standart hale getirildi.

### 4. Model Eğitimi ve Değerlendirme

- Modeller **k-fold çapraz doğrulama** ile eğitildi. k değeri olarak 10 kullanıldı.
-Eğitilen Modeller
  - Lojistik Regresyon (LR)
  - Karar Ağacı (DT)
  - K-En Yakın Komşuluk (KNN)
  - Naive Bayes (NB)
  - Destek Vektör Makineleri (SVM)
  - AdaBoost Sınıflandırıcı (AdaB)
  - Gradient Boosting Makinesi (GBM)
  - Rastgele Orman Sınıflandırıcısı (RF)
- Modellerin performansları, k-çapraz doğrulama doğruluk değerlerini kutu grafikleri ile karşılaştırıldı.

### 5. Hiperparametre Optimizasyonu

- En düşük değeri veren Karar Ağacı Sınıflandırıcısı için hiperparametreler **GridSearchCV** ile optimize edildi ve performansı arttırılmaya çalışıldı.
- `criterion`, `max_depth`, `min_samples_split` ve `min_samples_leaf` parametreleri incelendi.
- En iyi model, çapraz doğrulama doğruluğuyla seçildi.

### 6. Model Değerlendirme

- En iyi model test seti ile değerlendirildi:
  - Confusion Matrix
  - Sınıflandırma Raporu (Precision, Recall, F1-Score)

### 7. Gerçek Veri ile Test

- Eğitilmiş model, yeni veri girdileriyle test edildi ve tahminlerde bulunuldu.
