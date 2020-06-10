import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error

#1. csv 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
sunmission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

#1-1. 데이터 결측치 제거
# print(train.isnull().sum())
# train이 있는 데이터에 null값의 합계를 가져와라
train = train.interpolate()
train = train.fillna(train.mean())
# 컬럼별 보간법. 선형보간 (평타 85점) 옆에 컬럼에 영향 X
# 빈자리를 선에 맞게 그려준다
# 선형의 시작점이 nan이면 결측지 제거 X
# print(train.isnull().sum())
test = test.interpolate()
test = test.fillna(test.mean())

#1-2. 넘파이 불러오기
train = np.load('./data/dacon/comp1/train.npy')
test = np.load('./data/dacon/comp1/test.npy')
submission = np.load('./data/dacon/comp1/submission.npy')

#1-2. 데이터 자르기
x = train[:, :71]
y = train[:, -4:]
# print(x.shape)  # (10000, 71)
x_pred = test
# print(x_pred.shape) # (10000, 71)

#1-3. PCA
pca = PCA(n_components=60)
x = pca.fit_transform(x)
x_pred = pca.fit_transform(x_pred)
# print(x.shape)      # (10000, 60)
# print(x_pred.shape) # (10000, 60)

#1-4. RobustScaler
scaler = RobustScaler()
x = scaler.fit_transform(x)
x_pred = scaler.fit_transform(x_pred)

#1-5. LSTM 3차원 reshape
x = x.reshape(-1,1,60)
x_pred = x_pred.reshape(-1,1,60)

#1-6. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=99)

#2. 모델 구성
input1 = Input(shape=(1,60))
dense1 = LSTM(30, activation='relu', return_sequences=True)(input1)
dense1 = LSTM(40, activation='relu')(dense1)
dense1 = Dense(500, activation='relu')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(80, activation='relu')(dense1)
output1 = Dense(4, activation='relu')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'       
check_p = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, callbacks=[early_stop, check_p])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=10)
y_pred = model.predict(x_pred)
print(y_pred)
print("loss: ", loss)
print("mae: ", mae)
# loss:  1.760384011864662
# mae:  1.760384202003479

#5. submit할 파일 생성
y_pred = pd.DataFrame(y_pred, index=np.arange(10000,20000))
y_pred.to_csv('./data/dacon/comp1/submission_lstm.csv', header=["hhb","hbo2","ca","na"], index=True, index_label="id")
