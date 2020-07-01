import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

#1. 데이터
boston =  load_boston()
x = boston.data
y = boston.target
# print(x.shape)        # (506, 13)
# print(y.shape)        # (506,)

#1-1. 데이터 전처리
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x[1])

pca = PCA(n_components=8)
pca.fit(x)
x_low = pca.transform(x)
# print(x_low.shape)              # (506,8)

x = x_low.reshape(-1, 4, 2, 1)
# print(x.shape)                    # (506, 4, 2, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, train_size=0.6)

#2. 모델 구성
input1 = Input(shape=(4,2,1))
dense1 = Conv2D(32, (2,2), padding='same')(input1)
dense1 = Conv2D(64, (2,2), padding='same')(dense1)
dense1 = Conv2D(128, (2,2), padding='same')(dense1)
dense1 = Conv2D(256, (2,2), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=3, padding='same')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(224, (2,1), padding='same')(dense1)
dense1 = Conv2D(160, (2,1), padding='same')(dense1)
dense1 = Conv2D(69, (2,1), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2, padding='same')(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(10)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
earlystopping = EarlyStopping(monitor='mse', patience=5, mode='auto')
modelpath = './model/boston/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[earlystopping, checkpoint])

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)
# print("y_predict: \n", y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)
