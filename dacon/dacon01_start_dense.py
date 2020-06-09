import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error

#1. csv 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)
# print("train.shape: ", train.shape)         # (10000, 75) : x_train, x_test
# print("test.shape: ", test.shape)           # (10000, 71) : x_pred
# test.csv에는 y값이 될 컬럼이 존재하지 않으므로 x_pred로 지정
# print("submission.shape", submission.shape) # (10000, 4) : y_pred

#1-1. 데이터 결측치 제거
# print(train.isnull().sum())
# train이 있는 데이터에 null값의 합계를 가져와라
train = train.interpolate()
train = train.fillna(method='bfill')
# 컬럼별 보간법. 선형보간 (평타 85점) 옆에 컬럼에 영향 X
# 빈자리를 선에 맞게 그려준다
# 선형의 시작점이 nan이면 결측지 제거 X
# print(train.isnull().sum())
test = test.interpolate()
test = test.fillna(method='bfill')

#1-2. 데이터 자르기
x = train.iloc[:, :71]
y = train.iloc[:, -4:]
# print(x.shape)  # (10000, 71)
# print(y.shape)  # (10000, 4)
x_pred = test.iloc[:, :]
# print(x_pred.shape) # (10000, 71)

#1-3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=99)
# print(x_train.shape)    # (8000, 71)
# print(x_test.shape)     # (2000, 71)
# print(y_train.shape)    # (8000, 4)
# print(y_test.shape)     # (2000, 4)

#2. 모델 구성
input1 = Input(shape=(71, ))
dense1 = Dense(3000)(input1)
dense1 = Dense(5000)(input1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(7000)(dense1)
dense1 = Dense(4000)(dense1)
dense1 = Dense(1000)(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(100)(dense1)
output1 = Dense(4)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'       
check_p = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=5000, batch_size=50, validation_split=0.2, callbacks=[early_stop, check_p])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=50)
y_pred = model.predict(x_pred)
print(y_pred)
print("loss: ", loss)
print("mae: ", mae)
# loss:  1.7843858184814454
# mae:  1.7843859195709229

#5. submit할 파일 생성
y_pred = pd.DataFrame(y_pred, index=np.arange(10000,20000), columns=["hhb","hbo2","ca","na"])
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', header=0, index=True, index_label="id")
