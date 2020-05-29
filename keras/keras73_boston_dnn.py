import numpy as np
from sklearn.datasets import load_boston
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

#1. 데이터
boston =  load_boston()
x = boston.data
y = boston.target
# print(x.shape)        # (506, 13)
# print(y.shape)      # (506,)

#1-1. 데이터 전처리
y = y.reshape(y.shape[0],1)
# print(y.shape)                  # (506, 1)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x[1])
# print(x.shape)                  # (506, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

#2. 모델 구성
input1 = Input(shape=(13, ))
dense1 = Dense(100)(input1)
dense1 = Dense(5000)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(500)(dense1)
dense1 = Dense(300)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='mse', patience=5, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_mse', save_best_only=True, mode='auto')
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True, )
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[earlystopping, checkpoint, tb_hist])

#4. 평가, 예측
RMSE, r2 = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)
print("y_predict: \n", y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: \n", RMSE)

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2: \n", r2)
'''
RMSE = hist.history['RMSE']
r2 = hist.history['r2']
val_RMSE = hist.history['val_RMSE']
val_r2 = hist.history['val_r2']

print("val_RMSE: \n", val_RMSE)
print("val_r2: \n", val_r2)

#5. 시각화
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(RMSE, marker='.', c='red', label='RMSE')
plt.plot(val_RMSE, marker='.', c='blue', label='val_RMSE')
plt.grid()
plt.title('RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2,1,2)
plt.plot(r2, marker='.', c='red', label='r2')
plt.plot(val_r2, marker='.', c='blue', label='val_r2')
plt.grid()
plt.title('r2')
plt.ylabel('r2')
plt.xlabel('epoch')
plt.legend()
plt.show()
'''
