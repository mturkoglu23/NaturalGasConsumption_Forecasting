from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.optimizers import Adam
from hyperopt import hp, fmin, tpe, Trials,anneal
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('ngdataset-consumption.xlsx')
data['DATE'] = pd.to_datetime(data['DATE'], format='%d.%m.%Y')

data['Day'] = data['DATE'].dt.day
data['Month'] = data['DATE'].dt.month
data['Year'] = data['DATE'].dt.year

x_cols= ['Month','Year','EPU','GEA','DXY','HNGSP','IPI','NGP',	'REP','WTI','CPI']
y_col = ['NGC']

X = data[x_cols].replace({',': '.'}, regex=True).astype(float)
y = data[y_col].replace({',': '.'}, regex=True).astype(float)

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
X = scaler_x.fit_transform(X)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)
# y=y.values.ravel()

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

def cnn_model(params):
    cnn_scores = []
    for train_index, test_index in kfold.split(X):
        X_train_fold, X_test_fold = np.expand_dims(X[train_index], axis=2), np.expand_dims(X[test_index], axis=2)
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        cnn_model = Sequential()
        for i in range(params['conv_layers']):
            cnn_model.add(Conv1D(filters=params[f'filters_{i}'], kernel_size=params['kernel_size'], activation='relu', input_shape=(X.shape[1], 1)))
        cnn_model.add(Flatten())
        for i in range(params['dense_layers']):
            cnn_model.add(Dense(params[f'dense_units_{i}'], activation='relu'))
        cnn_model.add(Dense(1))
        cnn_model.compile(optimizer=Adam(), loss='mse')

        cnn_model.fit(X_train_fold, y_train_fold, epochs=1000, verbose=0)
        y_test_pred_fold = cnn_model.predict(X_test_fold)
        mse_fold = mean_squared_error(y_test_fold, y_test_pred_fold)
        cnn_scores.append(mse_fold)
    return np.mean(cnn_scores)

def create_search_space(conv_layers_range, filters_range, dense_layers_range, dense_units_range):
    space = {
        'conv_layers': hp.choice('conv_layers', conv_layers_range),
        'kernel_size': hp.choice('kernel_size', [2]),
        'dense_layers': hp.choice('dense_layers', dense_layers_range)
    }
    for i in range(max(conv_layers_range)):
        space[f'filters_{i}'] = hp.choice(f'filters_{i}', filters_range)
    for i in range(max(dense_layers_range)):
        space[f'dense_units_{i}'] = hp.choice(f'dense_units_{i}', dense_units_range)
    return space

conv_layers_range = [1, 2, 3,4,5]
filters_range = list(range(4, 257))
dense_layers_range = [1, 2, 3,4,5]
dense_units_range = list(range(10, 401))

space = create_search_space(conv_layers_range, filters_range, dense_layers_range, dense_units_range)

trials = Trials()
best = fmin(fn=cnn_model, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

print("Best Parameters:")
print(best)
