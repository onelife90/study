import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input
import matplotlib.pyplot as plt

#1. csv 불러오기
train = pd.read_csv('./data/dacon/comp3/train_features.csv', header=0, index_col=0)
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', header=0, index_col=0)
submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv', header=0, index_col=0)

#1-1. shape 확인
# id당 375행의 데이터
# print(train.shape)          # (1050000, 5)==(2800*375, 5)
# print(train_target.shape)   # (2800, 4)
# print(test.shape)           # (262500, 5)==(700*375, 5)
# print(submit.shape)         # (700, 4)

#1-2. 결측치 확인
# print(train.isnull().sum())         # 결측치 X
# print(train_target.isnull().sum())  # 결측치 X
# print(test.isnull().sum())          # 결측치 X

#1-3. 시계열 확인
# print(train.head()) 
# 일정한 시간 간격으로 가속도 측정
#    id      Time   S1   S2   S3   S4
# 0   0  0.000000  0.0  0.0  0.0  0.0
# 1   0  0.000004  0.0  0.0  0.0  0.0
# 2   0  0.000008  0.0  0.0  0.0  0.0
# 3   0  0.000012  0.0  0.0  0.0  0.0
# 4   0  0.000016  0.0  0.0  0.0  0.0

#1-4. 넘파이 저장
train = train.values
test = test.values
y = train_target.values
np.save('./data/dacon/comp3/train_features.npy', arr=train)
np.save('./data/dacon/comp3/test_features.npy', arr=test)
np.save('./data/dacon/comp3/train_target.npy', arr=y)

#1-5. id 컬럼 제외 데이터 슬라이싱
x = train[:, 1:]
x_pred = test[:, 1:]

#1-6. Scaler 후 reshape
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_pred = scaler.fit_transform(x_pred)

x = x.reshape(2800,375,4,1)
x_pred = x_pred.reshape(700,375,4,1)

#1-7. train_test_split
x_train, x_test, y_train, y_test = tts(x,y, random_state=88, test_size=0.2)

#2. 모델 구성
input1 = Input(shape=(375,4,1))
dense1 = Conv2D(175, (3,3), padding='same')(input1)
dense1 = Conv2D(75, (2,2), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2, padding='same')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(75, (2,2), padding='same')(dense1)
dense1 = Conv2D(15, (2,2), padding='same')(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(4)(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
y_pred = model.predict(x_pred)
print("y_pred : \n", y_pred)
print("mse: ", mse)

#5. submit할 csv 파일 생성
y_pred = pd.DataFrame(y_pred, index=np.arange(2800,3500))
y_pred.to_csv('./data/dacon/comp3/submit_Conv2D.csv', header=["X","Y","M","V"], index=True, index_label="id")
