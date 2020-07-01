import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# diabets 당뇨병. 회귀모델

#1. 데이터
diabets = load_diabetes()
x = diabets.data
y = diabets.target
# print(x[0])         # 10개 컬럼 
# print(y)            # 여러가지 값. 회귀모델
# print(x.shape)      # (442,10)
# print(y.shape)      # (442, )

#1-1. 데이터 전처리
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x[0])      

x = x.reshape(-1,5,2,1)
# print(x.shape)          # (442,5,2,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, train_size=0.8)

#2. 모델
input1 = Input(shape=(5,2,1))
dense1 = Conv2D(10, (2,1), padding='same')(input1)
dense1 = Conv2D(20, (2,1), padding='same')(dense1)
dense1 = Conv2D(40, (2,1), padding='same')(dense1)
dense1 = Conv2D(60, (2,1), padding='same')(dense1)
dense1 = Conv2D(80, (2,1), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2, padding='same')(dense1)
dense1 = Conv2D(70, (2,1), padding='same')(dense1)
dense1 = Conv2D(50, (2,1), padding='same')(dense1)
dense1 = Conv2D(30, (2,1), padding='same')(dense1)
dense1 = Conv2D(10, (2,1), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2, padding='same')(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/diabets/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=5000, batch_size=1, validation_split=0.2, callbacks=[early_stop, checkpoint])

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
y_pred = model.predict(x_test)

#RMSE 구하기
def RMSE(mean_y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", RMSE(y_test, y_pred))

#R2 구하기
r2 = r2_score(y_test, y_pred)
print("R2: ", r2)
