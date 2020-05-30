import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. 데이터
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# print(x[0])         # 30개 컬럼
# print(y[0])         # 예측모델..? 회귀모델
# print(x.shape)      # (569,30)
# print(y.shape)      # (569, )

#1-1. 데이터 전처리
y = y.reshape(-1,1)
# print(y.shape)      # (569,1)

scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)      # (569,30)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, train_size=0.6)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(30, )))
model.add(Dense(5000))
model.add(Dense(9000))
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

#RMSE 구하기
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

#R2 구하기
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
plt.plot(mse, marker='.', c='red', label='mse')
plt.plot(val_mse, marker='.', c='blue', label='val_mse')
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()
plt.show()
