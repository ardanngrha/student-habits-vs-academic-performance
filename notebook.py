#!/usr/bin/env python
# coding: utf-8

# ### Import Library yang dibutuhkan

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# # Data Understanding

# ##### Gathering Data

# In[2]:


df = pd.read_csv("student_habits_performance.csv")


# In[3]:


df.sample(5)


# ##### Assessing Data

# In[4]:


df.info()


# In[5]:


df.isna().sum()


# Dataset menunjukan ada 91 nilai null pada kolom `parental_education_level`.

# In[6]:


df[df.isna().any(axis=1)]


# In[7]:


df["parental_education_level"].describe()


# ##### Cleaning Data

# Menghapus kolom yang tidak perlu di dataset yaitu `student_id` dan mengisi missing value di kolom `parental_education_level` dengan modus dari kolom tersebut (High School)

# In[8]:


# hapus kolom student_id
df.drop(columns=["student_id"], inplace=True)


# In[9]:


most_frequent = df['parental_education_level'].mode()[0]
df['parental_education_level'] = df['parental_education_level'].fillna(most_frequent)


# In[10]:


df.info()


# In[11]:


df.isna().sum()


# Pengecekan jumlah missing value menunjukan bahwa sudah tidak ada missing value di dataset

# ### Exploratory Data Analysis (EDA)

# Pengecekan data statistik variabel numerik

# In[12]:


df.describe()


# In[13]:


categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                    'internet_quality', 'extracurricular_participation']


# Pengecekan data statistik variabel kategorikal

# In[14]:


df[categorical_cols].describe()


# Visualisasi distribusi data variabel numerik

# In[15]:


# Histogram semua kolom numerik
df.hist(bins=20, figsize=(20, 15))
plt.show()


# Visualisasi distribusi kolom kategorikal

# In[16]:


fig, axes = plt.subplots(3, 2, figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    ax = axes[i // 2, i % 2]
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f'Distribusi {col}')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


# In[17]:


print(f"Nilai minimum: {df['exam_score'].min()}")
print(f"Nilai maksimum: {df['exam_score'].max()}")
print(f"Nilai rata-rata: {df['exam_score'].mean():.2f}")
print(f"Standar deviasi: {df['exam_score'].std():.2f}")


# In[18]:


# Analisis Korelasi antara kolom Numerik
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation = df[numeric_cols].corr()

plt.figure(figsize=(9, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Kolom Numerik')
plt.tight_layout()
plt.show()


# Heatmap korelasi menunjukan bahwa:
# 
# - Variabel seperti `study_hours_per_day`, `sleep_hours`, `exercise_frequency`, dan `mental_health_rating` memiliki korelasi positif dengan `exam_score`
# - Variabel `study_hours_per_day` sangat berpengaruh positif dengan variabel `exam_score`
# - Variabel seperti `social_media_hours` dan `netflix_hours` memiliki korelasi negatif dengan `exam_score`

# In[19]:


categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                    'internet_quality', 'extracurricular_participation']


# In[20]:


# Analisis Hubungan antara Jam Belajar dan Nilai Ujian karena yang sangat berpengaruh
plt.figure(figsize=(9, 6))
sns.scatterplot(x='study_hours_per_day', y='exam_score', data=df)
plt.title('Hubungan antara Jam Belajar per Hari dan Nilai Ujian')
plt.xlabel('Jam Belajar per Hari')
plt.ylabel('Nilai Ujian')
plt.show()


# Dari hasil scatterplot, dapat disimpulkan bahwa semakin banyak jam belajar per hari, semakin tinggi nilai ujian siswa

# # Data Preparation

# Pada tahap data preparation, tidak dilakukan penghilangan outlier dikarenakan data yang digunakan memang mewakili kasus nyata yang sangat bervariasi

# In[21]:


df.head()


# Untuk variabel kategorikal nominal seperti `gender` dan `internet_quality` karena tidak ada hubungan ordinal (tingkatan) antar kategori, Untuk variabel `diet_quality` yang bersifat ordinal (memiliki tingkatan), kita dapat menggunakan Ordinal Encoding

# In[22]:


# Encoding Variabel Kategorikal
# One-Hot Encoding untuk variabel kategorik
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# In[23]:


# Ordinal Encoding untuk variabel diet_quality
diet_mapping = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
df['diet_quality_encoded'] = df['diet_quality'].map(diet_mapping)


# In[24]:


df_encoded.head()


# In[25]:


# Feature Scaling
# Normalisasi fitur numerik
from sklearn.preprocessing import StandardScaler

numeric_features = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                     'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
                     'mental_health_rating']

scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])


# In[26]:


df_encoded.head()


# Hanya memilih fitur-fitur yang berpengaruh untuk meningkatkan performa model dan menghindari overfitting.

# In[27]:


# Feature Selection
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


# Membagi data dengan rasio 80:20, 80% untuk training dan 20% untuk testing.

# In[28]:


# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

print(f"Jumlah sampel training: {X_train.shape[0]}")
print(f"Jumlah sampel testing: {X_test.shape[0]}")


# # Modelling

# **Linear Regression**

# In[29]:


from sklearn.linear_model import LinearRegression

# Membuat dan melatih model Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# **Random Forest Regressor**

# In[30]:


from sklearn.ensemble import RandomForestRegressor

# Membuat dan melatih model Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)


# **Gradient Boosting Regressor**

# In[31]:


from sklearn.ensemble import GradientBoostingRegressor

# Membuat dan melatih model Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)


# # Evaluation

# In[32]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[33]:


# Prediksi pada data testing
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)


# Metrik evaluasi yang digunakan adalah $MAE$, $MSE$, $RMSE$, dan $R^2$

# In[34]:


mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)


# In[35]:


# 4 Plot Perbandingan MAE, MSE, RMSE, dan R2 tiap Model
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MAE': [mae_lr, mae_rf, mae_gb],
    'MSE': [mse_lr, mse_rf, mse_gb],
    'RMSE': [rmse_lr, rmse_rf, rmse_gb],
    'R²': [r2_lr, r2_rf, r2_gb]
})
results.set_index('Model', inplace=True)
results.sort_values(by='RMSE', ascending=True, inplace=True)
print("\nPerbandingan Model:")
print(results)

results_transposed = results.T  # Transpose dataframe untuk menukar baris dan kolom

plt.figure(figsize=(12, 8))
results_transposed.plot(kind='bar')
plt.title('Perbandingan Model berdasarkan MAE, MSE, RMSE, dan R²')
plt.xlabel('Jenis Metrik')
plt.ylabel('Nilai')
plt.xticks(rotation=0)  # Rotasi x-axis labels ke horizontal
plt.legend(title='Algoritma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Berdasarkan plot perbandingan model, urutan algoritma dari yang terbaik adalah **Linear Regression**, **Gradient Boosting Regressor**, dan **Random Forest Regressor**.

# In[36]:


# # Mencoba prediksi nilai ujian untuk siswa baru
new_student = [-2.41806844, -1.11406369, -0.66975012,  0.09242597,  1.24812012,  1.46116644,  0.90018637,  0,          1,          0       ] # Contoh data siswa baru

new_student = np.array(new_student).reshape(1, -1)  # Ubah bentuk menjadi 2D array

predicted_score = lr_model.predict(new_student)
print(f"Prediksi nilai ujian untuk siswa baru: {predicted_score[0]:.2f}")


# In[ ]:




