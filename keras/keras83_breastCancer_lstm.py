import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

#1. 데이터
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# print(x[0])         # 30개 컬럼
# print(y[0])         # 0,1 이진분류
# print(x.shape)      # (569,30)
# print(y.shape)      # (569, )

x = x.reshape(-1,6,5)
print(x.shape)      # (596,6,5)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, train_size=0.6)

#2. 모델 구성
model = Sequential()
model.add(LSTM(50, input_shape=(6,5)))
model.add(Dense(9000))
model.add(Dense(5000))
model.add(Dropout(0.1))
model.add(Dense(7000))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,  write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[early_stop, checkpoint, tb_hist])

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=5)
y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

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
plt.plot(mse, marker='.', c='red', label='acc')
plt.plot(val_mse, marker='.', c='blue', label='val_mse')
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend()
plt.show()

# 튜닝
# epochs=22,batch=5,노드=LSTM50,9000,5000,drop0.1,7000,300,100
#RMSE:  <function RMSE at 0x0000024BDFD3F168>
#r2:  0.6934145390683433
