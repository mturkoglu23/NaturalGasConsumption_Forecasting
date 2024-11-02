import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler

# Veri yükleme
data = pd.read_excel('ngdataset-consumption.xlsx')
data['DATE'] = pd.to_datetime(data['DATE'], format='%d.%m.%Y')

# Gün, Ay ve Yıl sütunlarını oluştur
data['Day'] = data['DATE'].dt.day
data['Month'] = data['DATE'].dt.month
data['Year'] = data['DATE'].dt.year

x_cols = ['Month', 'Year', 'GEA', 'DXY', 'HNGSP', 'IPI', 'NGP', 'REP']
y_col = ['NGC']

# X ve Y verilerini ayır
X = data[x_cols].replace({',': '.'}, regex=True).astype(float)
y = data[y_col].replace({',': '.'}, regex=True).astype(float)

# Verileri normalleştirme
scaler_x = MinMaxScaler()
X = scaler_x.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# Sklearn modelleri
models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor(),
    "GBR": GradientBoostingRegressor()
}

for name, model in models.items():
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)  # RMSE'ye dönüştürmek için negatif MSE'yi pozitif RMSE'ye çeviririz
    mae_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    mae_scores = -mae_scores  # Negatif MAE'yi pozitif MAE'ye dönüştürmek için işaret ters çevrilir
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

    print(name + ":")
    print("Mean Absolute Error (MAE):", mae_scores.mean())
    print("Mean Squared Error (MSE):", scores.mean() ** 2)
    print("Root Mean Squared Error (RMSE):", scores.mean())
    print("R-squared (R2):", r2_scores.mean())
    print("\n")

# 1DCNN Model
X_cnn = np.expand_dims(X, axis=2)  # Boyut eklemek için
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=130, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)))
cnn_model.add(Conv1D(filters=219, kernel_size=2, activation='relu'))
cnn_model.add(Flatten())
cnn_model.add(Dense(44, activation='relu'))
cnn_model.add(Dense(171, activation='relu'))
cnn_model.add(Dense(274, activation='relu'))
cnn_model.add(Dense(1))
cnn_model.compile(optimizer='adam', loss='mse')

cnn_model.summary()

# 10 kat çapraz doğrulama ile eğitim ve değerlendirme
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cnn_rmse_scores = []
cnn_mae_scores = []
cnn_r2_scores = []
cnn_mse_scores = []
y_test_list = []
y_cnn_head_list = []

for train_index, test_index in kfold.split(X_cnn):
    X_train_fold, X_test_fold = X_cnn[train_index], X_cnn[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    cnn_model.fit(X_train_fold, y_train_fold, epochs=1000, verbose=1)
    y_test_pred_fold = cnn_model.predict(X_test_fold)

    mse_fold = mean_squared_error(y_test_fold, y_test_pred_fold)
    rmse_fold = np.sqrt(mse_fold)
    mae_fold = mean_absolute_error(y_test_fold, y_test_pred_fold)
    r2_fold = r2_score(y_test_fold, y_test_pred_fold)
    
    cnn_rmse_scores.append(rmse_fold)
    cnn_mae_scores.append(mae_fold)
    cnn_r2_scores.append(r2_fold)
    cnn_mse_scores.append(mse_fold)
    
    y_test_list.append(y_test_fold)
    y_cnn_head_list.append(y_test_pred_fold)

print("1D CNN:")
print("Mean Absolute Error (MAE):", np.mean(cnn_mae_scores))
print("Mean Squared Error (MSE):", np.mean(cnn_mse_scores))
print("Root Mean Squared Error (RMSE):", np.mean(cnn_rmse_scores))
print("R-squared (R2):", np.mean(cnn_r2_scores))
