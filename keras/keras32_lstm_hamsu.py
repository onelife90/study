from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([50,60,70])

print("x.shape : ", x.shape)        # (13,3)
print("y.shape : ", y.shape)        # (13,)
print("x_predict.shape : ", x_predict.shape)       # (3,)

x = x.reshape(x.shape[0], x.shape[1], 1)   
x_predict = x_predict.reshape(1,3,1)

print(x.shape)          #(13,3,1)
print(x_predict.shape)  #(1,3,1)

#2. 모델구성

input1 = Input(shape=(3,1))
dense1 = LSTM(5)(input1)
dense1 = Dense(10)(dense1)
dense1 = Dense(7)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

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
# epochs=900 노드=3,5,10,71
# [[72.66392]
