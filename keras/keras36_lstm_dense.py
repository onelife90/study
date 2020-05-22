from numpy import array
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([55,65,75])

print("x.shape : ", x.shape)        # (13,3)
print("y.shape : ", y.shape)        # (13,)
print("x_predict.shape : ", x_predict.shape)       # (3,)

x_predict = x_predict.reshape(1,3)
print(x_predict.shape)  #(1,3)

#2. 모델구성

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x,y, epochs=5000, callbacks=[early_stopping])

print(x_predict)

#4. 평가, 예측
y_predict = model.predict(x_predict)
print(y_predict)

loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)

# 하이퍼파라미터 튜닝
# epochs=400 노드=7,5,3,2,3,4,3,1
# [[84.999985]] / [[107.025955]]

# epochs=8 노드=3,6,8,4000,5,4,3,1
# [[81.934]] / [[96.3795]]

# epochs=9 노드=3,6,80,1000,5,4,3,1
# [[89.35123]]

# epochs=12 노드=3,5,15,35,25,15,1
# [[88.86557]]

# epochs= 노드=1,20,20,20,5000,20,20,2000,20,20,20,20,1
# [[88.267166]]
