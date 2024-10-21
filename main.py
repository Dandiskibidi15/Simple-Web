# Mengimpor pustaka yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca dataset
data = pd.read_csv('data.csv')

# Melihat 5 baris pertama data
print(data.head())

# Deskripsi statistik dasar
print(data.describe())

# Visualisasi data
plt.figure(figsize=(10,6))
plt.scatter(data['X'], data['Y'], color='blue')
plt.title('Scatter plot of X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Membagi data ke dalam set pelatihan dan pengujian
X = data[['X']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi nilai
y_pred = model.predict(X_test)

# Metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot hasil prediksi vs nilai sebenarnya
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='True Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.legend()
plt.title('True vs Predicted Values')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
