from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Dense, Input, Concatenate
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
housing = fetch_california_housing()

x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

scaler = StandardScaler()
x_train - scaler.fit_transform(x_train)
x_val - scaler.fit_transform(x_val)
x_test - scaler.fit_transform(x_test)

# 회귀 : 출력층이 활성화 함수가 없는 하나의 뉴런(하나의 값을 예측하기 때문), 손실함수로 평균 제곱 오차 사용
#2. 모델 구성
input1 = Input(shape=x_train.shape[1:])
hide1 = Dense(30, activation='relu')(input1)
hide1 = Dense(30, activation='relu')(hide1)
concat = Concatenate()([input1, hide1])
output1 = Dense(1)(concat)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='sgd')
model.fit(x_train,y_train, epochs=20, validation_data=(x_val,y_val))

#4. 평가, 예측
mse_test = model.evaluate(x_test,y_test)
x_pred = x_test[:3]
y_pred = model.predict(x_pred)
