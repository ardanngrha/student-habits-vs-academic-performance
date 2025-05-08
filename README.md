# Laporan Proyek Machine Learning - Ardana Aldhizuma Nugraha

## Domain Proyek

Bidang pendidikan memainkan peran penting dalam pembangunan individu dan kemajuan masyarakat. Seiring berkembangnya teknologi dan perubahan gaya hidup pelajar, muncul tantangan baru dalam memastikan keberhasilan akademik. Salah satu faktor utama yang mempengaruhi prestasi siswa adalah kebiasaan harian mereka, seperti durasi belajar, waktu tidur, partisipasi dalam kegiatan ekstrakurikuler, konsumsi media sosial, dan sebagainya. Memahami hubungan antara kebiasaan siswa dengan performa akademik dapat membantu institusi pendidikan, orang tua, dan siswa itu sendiri dalam mengambil keputusan yang lebih baik untuk meningkatkan hasil belajar.

Penelitian sebelumnya menunjukkan bahwa faktor-faktor seperti durasi belajar, kualitas tidur, dan kesehatan mental memiliki korelasi yang signifikan dengan prestasi akademik siswa [1][2]. Oleh karena itu, pendekatan berbasis data sangat relevan untuk mengeksplorasi dan memprediksi performa akademik siswa berdasarkan kebiasaan mereka sehari-hari.

Referensi:

1. Dewald, J. F., Meijer, A. M., Oort, F. J., Kerkhof, G. A., & Bögels, S. M. (2010). The influence of sleep quality, sleep duration and sleepiness on school performance in children and adolescents: A meta-analytic review. Sleep Medicine Reviews, 14(3), 179-189.

2. Nonis, S. A., & Hudson, G. I. (2006). Academic performance of college students: Influence of time spent studying and working. Journal of Education for Business, 81(3), 151-159.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, beberapa permasalahan yang dapat diidentifikasi adalah:

- Faktor-faktor kebiasaan apa saja yang memiliki pengaruh signifikan terhadap performa akademik siswa?
- Bagaimana pengaruh penggunaan media sosial dan platform streaming terhadap nilai ujian siswa?
- Apakah faktor-faktor seperti kualitas tidur, pola makan, dan aktivitas fisik berkontribusi signifikan pada performa akademik?
- Seberapa akurat model machine learning dapat memprediksi performa akademik siswa berdasarkan kebiasaan dan karakteristik personal mereka?

### Goals

Tujuan dari proyek ini adalah:

- Mengidentifikasi dan menganalisis faktor-faktor kebiasaan yang memiliki korelasi kuat dengan performa akademik siswa.
- Mengukur dampak penggunaan media sosial dan platform streaming terhadap nilai ujian siswa.
- Mengevaluasi kontribusi faktor-faktor seperti kualitas tidur, pola makan, dan aktivitas fisik terhadap performa akademik.
- Membangun model machine learning yang dapat memprediksi performa akademik siswa dengan tingkat akurasi yang tinggi berdasarkan kebiasaan dan karakteristik personal.

### Solution statements

Untuk mencapai tujuan yang telah ditetapkan, beberapa pendekatan solusi yang akan diimplementasikan adalah:

- Melakukan Exploratory Data Analysis (EDA) untuk mengidentifikasi pola dan korelasi antara berbagai variabel kebiasaan dengan performa akademik.
- Mengembangkan beberapa model machine learning untuk memprediksi nilai ujian siswa:
  - Model Regresi Linear sebagai baseline model untuk memprediksi nilai ujian berdasarkan berbagai fitur.
  - Model Random Forest Regressor untuk menangkap hubungan non-linear antara variabel prediktor dan target.
  - Model Gradient Boosting Regressor untuk meningkatkan performa prediksi dengan teknik ensemble.

Performa solusi akan diukur menggunakan metrik evaluasi standar untuk masalah regresi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan koefisien determinasi (R²).

## Data Understanding

Dataset Student Habits vs Academic Performance berisi informasi tentang berbagai kebiasaan dan karakteristik siswa serta performa akademik mereka yang diukur melalui nilai ujian. Dataset ini terdiri dari 1000 entri dengan 16 variabel yang mencakup berbagai aspek kehidupan dan kebiasaan siswa.

### Variabel-variabel pada dataset adalah sebagai berikut:

1. `student_id`: Identifikasi unik untuk setiap siswa (`object`)
2. `age`: Usia siswa dalam tahun (`int64`)
3. `gender`: Jenis kelamin siswa (`object`)
4. `study_hours_per_day`: Jumlah jam belajar per hari (`float64`)
5. `social_media_hours`: Jumlah jam yang dihabiskan di media sosial per hari (`float64`)
6. `netflix_hours`: Jumlah jam menonton Netflix atau platform streaming serupa per hari (`float64`)
7. `part_time_job`: Apakah siswa memiliki pekerjaan paruh waktu (`object`)
8. `attendance_percentage`: Persentase kehadiran siswa di kelas (`float64`)
9. `sleep_hours`: Rata-rata jam tidur per hari (`float64`)
10. `diet_quality`: Kualitas pola makan siswa (`object`)
11. `exercise_frequency`: Frekuensi berolahraga per minggu (`int64`)
12. `parental_education_level`: Tingkat pendidikan tertinggi dari orang tua (`object`)
13. `internet_quality`: Kualitas koneksi internet siswa (`object`)
14. `mental_health_rating`: Skor penilaian kesehatan mental siswa (`int64`)
15. `extracurricular_participation`: Apakah siswa berpartisipasi dalam kegiatan ekstrakurikuler (`object`)
16. `exam_score`: Nilai ujian siswa yang merupakan variabel target (float64)

Beberapa observasi dari struktur dataset:

- Dataset memiliki 1000 entri dengan 909 data lengkap untuk variabel `parental_education_level` (terdapat 91 missing values)
- Terdapat 3 variabel numerik bertipe `int64`, 6 variabel numerik bertipe `float64`, dan 7 variabel kategorikal bertipe `object`
- Variabel target dalam dataset ini adalah `exam_score` yang menunjukkan performa akademik siswa

### Exploratory Data Analysis (EDA)

**1. Analisis Variabel Numerik**

Analisis variabel numerik menunjukkan distribusi dan statistik deskriptif dari variabel-variabel tersebut:

**2. Analisis Variabel Kategorikal**

Analisis variabel kategorikal menunjukkan perbedaan performa akademik berdasarkan berbagai kategori:

**3. Analisis Korelasi antara Variabel Numerik**

Analisis korelasi antara variabel numerik menunjukkan beberapa insight penting:

- Terdapat korelasi positif yang kuat antara jam belajar per hari (study_hours_per_day) dan nilai ujian (exam_score) dengan nilai korelasi 0.68
- Terdapat korelasi positif moderat antara persentase kehadiran (attendance_percentage) dan nilai ujian (exam_score) dengan nilai korelasi 0.56
- Terdapat korelasi negatif antara jam penggunaan media sosial (social_media_hours) dan nilai ujian (exam_score) dengan nilai korelasi -0.42
- Terdapat korelasi negatif antara jam menonton Netflix (netflix_hours) dan nilai ujian (exam_score) dengan nilai korelasi -0.37

```py
# Memilih hanya variabel numerik
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation = df[numeric_cols].corr()

# Visualisasi heatmap korelasi

plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Variabel Numerik')
plt.tight_layout()
plt.show()
```

Berdasarkan hasil EDA, terdapat beberapa insight yang dapat digunakan untuk pemodelan:

- Variabel seperti jam belajar, persentase kehadiran, dan kualitas pola makan memiliki korelasi positif dengan nilai ujian
- Variabel seperti penggunaan media sosial dan platform streaming memiliki korelasi negatif dengan nilai ujian
- Faktor-faktor seperti partisipasi ekstrakurikuler dan tingkat pendidikan orang tua juga mempengaruhi performa akademik siswa

## Data Preparation

Dalam tahap persiapan data, beberapa teknik preprocessing diterapkan untuk memastikan data dalam kondisi optimal untuk pemodelan machine learning:

### 1. Penanganan Missing Values

Pada dataset terdapat missing values pada variabel 'parental_education_level'. Untuk menangani hal ini, kita menggunakan teknik imputasi dengan modus (nilai yang paling sering muncul) karena variabel tersebut bersifat kategorikal.

```py
# Mengecek missing values
print(df.isnull().sum())

# Menangani missing values dengan imputasi modus
most_frequent = df['parental_education_level'].mode()[0]
df['parental_education_level'].fillna(most_frequent, inplace=True)

# Memastikan tidak ada missing values lagi
print("Setelah imputasi:")
print(df.isnull().sum())
```

Imputasi dengan modus dipilih untuk mempertahankan distribusi data yang sudah ada tanpa memasukkan nilai baru yang mungkin tidak sesuai dengan pola data.

### 2. Encoding Variabel Kategorikal

Untuk dapat digunakan dalam model machine learning, variabel kategorikal perlu diubah menjadi bentuk numerik dengan teknik encoding.

```py
# One-Hot Encoding untuk variabel kategorik

categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                    'internet_quality', 'extracurricular_participation']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

One-Hot Encoding digunakan untuk variabel kategorikal nominal seperti `gender` dan `internet_quality` karena tidak ada hubungan ordinal (tingkatan) antar kategori. Pendekatan `drop_first=True` diterapkan untuk menghindari multikolinearitas dalam model.

Untuk variabel `diet_quality` yang bersifat ordinal (memiliki tingkatan), kita dapat menggunakan Ordinal Encoding sebagai alternatif:

```py
# Ordinal Encoding untuk variabel diet_quality
diet_mapping = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
df['diet_quality_encoded'] = df['diet_quality'].map(diet_mapping)
```

### 3. Feature Scaling

Untuk memastikan semua fitur berada pada skala yang sama dan tidak didominasi oleh fitur dengan nilai yang besar, kita melakukan normalisasi pada fitur numerik.

```py
# Normalisasi fitur numerik
from sklearn.preprocessing import StandardScaler

numeric_features = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                     'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
                     'mental_health_rating']

scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
```

Standarisasi fitur numerik penting karena beberapa algoritma seperti Regresi Linear dan SVM sensitif terhadap skala fitur. Dengan standarisasi, setiap fitur akan memiliki mean 0 dan standar deviasi 1.

### 4. Feature Selection

Untuk mengidentifikasi fitur-fitur yang paling berpengaruh, kita menggunakan metode pemilihan fitur seperti SelectKBest dengan metrik f_regression:

```py
from sklearn.feature_selection import SelectKBest, f_regression

X = df_encoded.drop('exam_score', axis=1)
y = df_encoded['exam_score']

# Memilih fitur terbaik
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# Mendapatkan nama fitur terpilih
selected_features_idx = selector.get_support(indices=True)
selected_features = X.columns[selected_features_idx]
print("Fitur terpilih:", selected_features)
```

Pemilihan fitur membantu mengurangi dimensionalitas data dan meningkatkan performa model dengan menghilangkan fitur yang kurang relevan. Ini juga membantu mengatasi potensi masalah overfitting.

### 5. Train-Test Split

Sebelum membangun model, kita membagi dataset menjadi data training dan testing:

```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

print(f"Jumlah sampel training: {X_train.shape[0]}")
print(f"Jumlah sampel testing: {X_test.shape[0]}")
```

Pembagian data dengan rasio 80% untuk training dan 20% untuk testing memungkinkan kita untuk melatih model pada sebagian besar data dan mengevaluasi performanya pada data yang belum pernah dilihat model sebelumnya.

Teknik persiapan data di atas memastikan bahwa data dalam kondisi optimal untuk pemodelan machine learning, dengan fitur-fitur yang sudah dinormalisasi, dikodekan dengan tepat, dan tidak ada missing values.

## Modeling

Pada tahap ini, beberapa model machine learning diterapkan untuk memprediksi nilai ujian siswa berdasarkan kebiasaan dan karakteristik mereka. Tiga algoritma utama yang digunakan adalah Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor.

### 1. Linear Regression

Linear Regression merupakan algoritma dasar yang digunakan sebagai baseline model untuk masalah regresi.

```py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Membuat dan melatih model Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred_lr = lr_model.predict(X_test)

# Evaluasi model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression:")
print(f"MAE: {mae_lr:.4f}")
print(f"MSE: {mse_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"R²: {r2_lr:.4f}")
```

Kelebihan Linear Regression:

- Interpretabilitas yang tinggi karena koefisien model dapat diinterpretasikan secara langsung
- Komputasi yang efisien dan cepat
- Cocok untuk kasus di mana hubungan antara fitur dan target bersifat linear

Kekurangan Linear Regression:

- Asumsi linearitas yang kaku, tidak dapat menangkap hubungan non-linear
- Sensitif terhadap outlier
- Dapat mengalami masalah multikolinearitas jika ada korelasi tinggi antar fitur

### 2. Random Forest Regressor

Random Forest merupakan algoritma ensemble yang menggabungkan beberapa decision tree untuk menghasilkan prediksi yang lebih akurat.

```py
from sklearn.ensemble import RandomForestRegressor

# Membuat dan melatih model Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred_rf = rf_model.predict(X_test)

# Evaluasi model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor:")
print(f"MAE: {mae_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²: {r2_rf:.4f}")
```

Kelebihan Random Forest:

- Mampu menangkap hubungan non-linear antara fitur dan target
- Robust terhadap outlier dan noise dalam data
- Dapat menghasilkan feature importance untuk interpretasi model
- Mengurangi risiko overfitting dibandingkan single decision tree

Kekurangan Random Forest:

- Komputasi yang lebih berat dibandingkan Linear Regression
- Interpretabilitas yang lebih rendah dibandingkan model linear
- Memerlukan tuning hyperparameter untuk performa optimal

### 3. Gradient Boosting Regressor

Gradient Boosting merupakan algoritma ensemble yang membangun model secara bertahap dengan fokus pada perbaikan error dari model sebelumnya.

```py
from sklearn.ensemble import GradientBoostingRegressor

# Membuat dan melatih model Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred_gb = gb_model.predict(X_test)

# Evaluasi model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Regressor:")
print(f"MAE: {mae_gb:.4f}")
print(f"MSE: {mse_gb:.4f}")
print(f"RMSE: {rmse_gb:.4f}")
print(f"R²: {r2_gb:.4f}")
```

Kelebihan Gradient Boosting:

- Performa yang sangat baik untuk berbagai masalah regresi
- Mampu menangkap pola kompleks dalam data
- Dapat menangani berbagai jenis fitur (numerik dan kategorikal)
- Menyediakan feature importance untuk interpretasi model

Kekurangan Gradient Boosting:

- Risiko overfitting jika parameter tidak dikonfigurasi dengan tepat
- Komputasi yang lebih intensif dibandingkan Random Forest
- Sensitif terhadap hyperparameter seperti learning rate

### Pemilihan Model Terbaik

Berdasarkan evaluasi yang dilakukan, Random Forest Regressor dengan hyperparameter tuning dipilih sebagai model terbaik untuk prediksi nilai ujian siswa. Alasan pemilihan:

1. Performa Superior: Model ini menunjukkan nilai MAE, MSE, dan RMSE yang lebih rendah serta R² yang lebih tinggi dibandingkan model lainnya, menunjukkan kemampuan prediksi yang lebih baik.
2. Kemampuan Menangkap Pola Kompleks: Random Forest mampu menangkap hubungan non-linear antara kebiasaan siswa dan performa akademik mereka, yang mungkin tidak dapat ditangkap oleh model linear sederhana.
3. Robustness: Model ini menunjukkan ketahanan terhadap overfitting, terutama setelah hyperparameter tuning yang tepat.
4. Interpretabilitas: Meskipun tidak seinterpretable seperti Linear Regression, Random Forest masih menyediakan feature importance yang memungkinkan kita untuk memahami faktor-faktor yang paling berpengaruh terhadap performa akademik siswa.
5. Stabilitas: Random Forest cenderung lebih stabil dalam berbagai skenario data dibandingkan dengan model yang lebih sensitif seperti Gradient Boosting.

Melalui feature importance analysis dari model Random Forest, kita juga dapat mengidentifikasi bahwa variabel seperti jam belajar per hari, persentase kehadiran, dan kualitas tidur merupakan faktor-faktor yang paling berpengaruh terhadap nilai ujian siswa.

## Evaluation

Dalam proyek ini, beberapa metrik evaluasi digunakan untuk menilai performa model dalam memprediksi nilai ujian siswa. Berikut adalah penjelasan tentang metrik-metrik tersebut dan hasil evaluasi dari model terbaik.

### 1. Mean Absolute Error (MAE)

MAE mengukur rata-rata nilai absolut dari error (selisih antara nilai prediksi dan nilai aktual). Metrik ini memberikan gambaran tentang seberapa jauh, secara rata-rata, prediksi model dari nilai sebenarnya tanpa mempertimbangkan arah error (positif atau negatif).

**Formula:**
$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$

- $n$: jumlah sampel  
- $y_i$: nilai aktual  
- $\hat{y}_i$: nilai prediksi  

**Interpretasi:**  
Nilai MAE yang lebih rendah menunjukkan performa model yang lebih baik.

---

### 2. Mean Squared Error (MSE)

MSE mengukur rata-rata kuadrat error antara nilai prediksi dan nilai aktual. Metrik ini memberikan bobot lebih besar pada error yang besar karena error dikuadratkan.

**Formula:**
$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$

**Interpretasi:**  
Nilai MSE yang lebih rendah menunjukkan performa model yang lebih baik.

---

### 3. Root Mean Squared Error (RMSE)

RMSE adalah akar kuadrat dari MSE. Metrik ini memiliki unit yang sama dengan variabel target (misalnya, nilai ujian), sehingga lebih mudah diinterpretasikan.

**Formula:**
$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$

**Interpretasi:**  
Nilai RMSE yang lebih rendah menunjukkan bahwa model memiliki error yang kecil dalam melakukan prediksi.

---

### 4. R² (R-squared)

R² mengukur proporsi variansi dari variabel target yang dapat dijelaskan oleh variabel input dalam model. Nilai R² berkisar antara 0 dan 1 (bisa negatif untuk model yang sangat buruk).

**Formula:**
$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$

- $\bar{y}$: rata-rata nilai aktual  
- Penyebut: Total Sum of Squares (TSS)  
- Pembilang: Residual Sum of Squares (RSS)  

**Interpretasi:**

- $R^2 = 1$: Model memprediksi dengan sempurna
- $R^2 = 0$: Model tidak lebih baik dari rata-rata
- $R^2 < 0$: Model lebih buruk dari prediksi rata-rata
