import numpy as np
from sklearn.datasets import load_boston
from keras.utils import np_utils
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
y = y.reshape(y.shape[0],1)
# print(y.shape)                  # (503, 1)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x[1])
x = x.reshape(-1, x.shape[1], 1)
# print(x.shape)                  # (506, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)
print(x_train.shape)        # (303, 13, 1)                
print(x_test.shape)         # (203, 13, 1)
print(y_train.shape)        # (303, 1)
print(y_test.shape)         # (203, 1)

#2. 모델 구성
input1 = Input(shape=(13,1))
dense1 = LSTM(10, return_sequences=True)(input1)
dense1 = LSTM(1)(dense1)
dense1 = Dense(5000)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(500)(dense1)
dense1 = Dense(300)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
earlystopping = EarlyStopping(monitor='mse', patience=5, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True, )
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[earlystopping, checkpoint, tb_hist])

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

loss = hist.history['loss']
mse = hist.history['mse']
val_loss = hist.history['val_loss']
val_mse = hist.history['val_mse']

#5. 시각화
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2,1,2)
plt.plot(mse, marker='.', c='red', label='mse')
plt.plot(val_mse, marker='.', c='blue', label='val_mse')
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend()

plt.show()
