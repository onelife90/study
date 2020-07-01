import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. 데이터
boston =  load_boston()
x = boston.data
y = boston.target
# print(x.shape)        # (506, 13)
# print(y.shape)        # (506,)

#1-1. 데이터 전처리
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x[1])
x = x.reshape(-1, x.shape[1], 1)
# print(x.shape)                  # (506, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

#2. 모델 구성
input1 = Input(shape=(13,1))
dense1 = LSTM(13, return_sequences=True)(input1)
dense1 = LSTM(26)(dense1)
dense1 = Dense(52)(dense1)
dense1 = Dense(78)(dense1)
dense1 = Dense(104)(dense1)
dense1 = Dense(91)(dense1)
dense1 = Dense(65)(dense1)
dense1 = Dense(39)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
earlystopping = EarlyStopping(monitor='mse', patience=5, mode='auto')
modelpath = './model/boston/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[earlystopping, checkpoint])

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
y_pred = model.predict(x_test)

#RMSE 구하기
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", RMSE(y_test, y_pred))

#R2 구하기
r2 = r2_score(y_test, y_pred)
print("R2: ", r2)

# 튜닝
# epochs=100,batch=1,노드=LSTM10,LSTM1,5000,drop0.2,500,300
#RMSE:  7.926188563523864
#R2:  0.1622650037405542
