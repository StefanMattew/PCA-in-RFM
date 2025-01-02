import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Data Pre-processing
# DATASET = https://www.kaggle.com/code/alfathterry/customer-segmentation-rfm

file_path = "OnlineRetail.csv" 
data = pd.read_csv(file_path)

# Mengubah menjadi datetime contoh 02/12/2010
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate']) 
# Menghapus baris atau isi dari data yang hilang atau Nan
data = data.dropna(subset=['CustomerID']) 

# Hitung Monetary (Total Pendapatan per Baris)
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']


reference_date = datetime(2011, 12, 10)  #tanggal untuk acuan 

# Hitung RFM
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days, 
    'InvoiceNo': 'nunique',                             
    'TotalPrice': 'sum'                                      
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Skor R
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int) # menggunakan pd.qcut supaya lebih efisien

plt.subplot(1, 3, 1)
plt.bar(rfm['R_Score'].value_counts().index, rfm['R_Score'].value_counts().values)
plt.xlabel('R Score (Recency)')
plt.ylabel('Number of Customers')
plt.title('R Score Distribution')

# Skor F
rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=6, labels=False, duplicates='drop') + 1

plt.subplot(1, 3, 2)
plt.bar(rfm['F_Score'].value_counts().index, rfm['F_Score'].value_counts().values)
plt.xlabel('F Score (Recency)')
plt.ylabel('Number of Customers')
plt.title('F Score Distribution')

# Skor M
rfm['M_Score'] = rfm['Monetary'].rank(method='dense', ascending=True).apply(
    lambda x: min(int((x / len(rfm)) * 5) + 1, 5)
).astype(int)
plt.subplot(1, 3, 3)
plt.bar(rfm['M_Score'].value_counts().index, rfm['M_Score'].value_counts().values)
plt.xlabel('M Score (Recency)')
plt.ylabel('Number of Customers')
plt.title('M Score Distribution')


R_matrix = rfm['R_Score'].value_counts().sort_index()
F_matrix = rfm['F_Score'].value_counts().sort_index()
M_matrix = rfm['M_Score'].value_counts().sort_index()

# Menghitung rata rata R,F,dan M
rfm['RFM_Score'] = (rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']) / 3

# PCA
# Melakukan standardisasi terlebih dahulu
rfm_scaled = (rfm[['Recency', 'Frequency', 'Monetary']] - \
              rfm[['Recency', 'Frequency', 'Monetary']].mean()) / rfm[['Recency', 'Frequency', 'Monetary']].std()

# Matriks kovarina dengan numpy
cov_matrix = np.cov(rfm_scaled.T)

# Mengitung eigenvalue dan eigenvector
eigenvalue, eigenvector = np.linalg.eig(cov_matrix)


pca_result = np.dot(rfm_scaled, eigenvector[:, :2])
rfm['PCA1'] = pca_result[:, 0]
rfm['PCA2'] = pca_result[:, 1]

#Menampilkan hasil PCA 10 teratas berdasarkan customer id
print("Hasil PCA :")
print(rfm[['CustomerID', 'PCA1', 'PCA2']].head(10))

#Menampilkan hasil skor dari customer
print("Jumlah CustomerID untuk setiap skor R:")
print(R_matrix)
print()
print("Jumlah CustomerID untuk setiap skor F:")
print(F_matrix)
print()
print("Jumlah CustomerID untuk setiap skor M:")
print(M_matrix)
print()

#Menampilkan skor RFM per customer
print("RFM Scores per Customer:")
print(rfm[['CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']])
print()

# Rata-rata skor RFM
print("Rata-rata skor RFM:")
print(f"Rata-rata R: {rfm['R_Score'].mean():.2f}")
print(f"Rata-rata F: {rfm['F_Score'].mean():.2f}")
print(f"Rata-rata M: {rfm['M_Score'].mean():.2f}")
print(f"Rata-rata RFM: {rfm['RFM_Score'].mean():.2f}")

#Menuliskan hasil pca dan rfm ke dalam  file CSV
rfm.to_csv('rfm_pca.csv', index=False)
print("File RFM dengan PCA telah disimpan sebagai 'rfm_pca.csv'")

plt.tight_layout()
plt.show()

# s1 = rfm['M_Score'].value_counts()[5]
# print(f"m5: {s1}")

# S1
count_s1 = rfm[(rfm['M_Score'] == 5) & (rfm['F_Score'] == 5) & (rfm['R_Score'] == 5)]['M_Score'].count()
print(f"S1 : {count_s1}")

# S2
count_s2 = rfm[(rfm['M_Score'] == 4) & (rfm['F_Score'] == 4) & (rfm['R_Score'] == 4)]['M_Score'].count()
print(f"S2 : {count_s2}")

# S3
count_s3 = rfm[(rfm['M_Score'] == 3) & (rfm['F_Score'] == 4) & (rfm['R_Score'] == 5)]['M_Score'].count()
print(f"S3 : {count_s3}")

# S4
count_s4 = rfm[(rfm['M_Score'] == 3) & (rfm['F_Score'] == 2) & (rfm['R_Score'] == 5)]['M_Score'].count()
print(f"S4 : {count_s4}")

# S5
count_s5 = rfm[(rfm['M_Score'] == 3) & (rfm['F_Score'] == 3) & (rfm['R_Score'] == 3)]['M_Score'].count()
print(f"S5 : {count_s5}")

# S6
count_s6 = rfm[(rfm['M_Score'] == 4) & (rfm['F_Score'] == 3) & (rfm['R_Score'] == 2)]['M_Score'].count()
print(f"S6 : {count_s6}")

# S7
count_s7 = rfm[(rfm['M_Score'] == 2) & (rfm['F_Score'] == 2) & (rfm['R_Score'] == 2)]['M_Score'].count()
print(f"S7 : {count_s7}")

# S8
count_s8 = rfm[(rfm['M_Score'] == 2) & (rfm['F_Score'] == 4) & (rfm['R_Score'] == 5)]['M_Score'].count()
print(f"S8 : {count_s8}")

# S9
count_s9 = rfm[(rfm['M_Score'] == 4) & (rfm['F_Score'] == 4) & (rfm['R_Score'] == 2)]['M_Score'].count()
print(f"S9 : {count_s9}")

# S10
count_s10 = rfm[(rfm['M_Score'] == 2) & (rfm['F_Score'] == 2) & (rfm['R_Score'] == 2)]['M_Score'].count()
print(f"S10 : {count_s10}")

# S11
count_s11 = rfm[(rfm['M_Score'] == 1) & (rfm['F_Score'] == 1) & (rfm['R_Score'] == 1)]['M_Score'].count()
print(f"S11 : {count_s11}")

