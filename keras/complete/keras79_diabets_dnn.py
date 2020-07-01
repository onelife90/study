import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# diabets 당뇨병 회귀모델

#1. 데이터
diabets = load_diabetes()
x = diabets.data
y = diabets.target
# print(x[0])         # 10개 컬럼 
# print(y)            # 데이터셋 여러가지 값. 회귀모델
# print(x.shape)      # (442,10)
# print(y.shape)      # (442, )

#1-1. 데이터 전처리
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x[0])      

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, train_size=0.8)

#2. 모델
input1 = Input(shape=(10, ))
dense1 = Dense(10)(input1)
dense1 = Dense(20)(dense1)
dense1 = Dense(40)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(80)(dense1)
dense1 = Dense(70)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(30)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/diabets/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stop, checkpoint])

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
y_pred = model.predict(x_test)

#RMSE 구하기
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", RMSE(y_test, y_pred))

#R2 구하기
r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

# 튜닝
# epochs=22,batch=1,노드=9000,drop0.2,2340,442
#RMSE:  56.223379372889966
#r2:  0.47656005925434797
