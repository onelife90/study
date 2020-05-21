from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70], [60,70,80], [70,80,90], [80,90,100],
           [90,100,110], [100,110,120],
           [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])            
x2_predict = array([65,75,85])            

print("x.shape : ", x1.shape)        # (13,3)
print("x.shape : ", x2.shape)        # (13,3)
print("y.shape : ", y.shape)         # (13,)
print("x_predict.shape : ", x1_predict.shape)       # (3,)
print("x_predict.shape : ", x2_predict.shape)       # (3,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)   
x2 = x2.reshape(x2.shape[0], x1.shape[1], 1)   
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

print(x1.shape)          #(13,3,1)
print(x2.shape)          #(13,3,1)
print(x1_predict.shape)  #(1,3,1)
print(x2_predict.shape)  #(1,3,1)

#2. 모델구성

input1 = Input(shape=(3,1))
dense1 = LSTM(10)(input1)
dense1 = Dense(4)(dense1)
dense1 = Dense(3)(dense1)
dense1 = Dense(1)(dense1)

input2 = Input(shape=(3,1))
dense2 = LSTM(10)(input2)
dense2 = Dense(4)(dense2)
dense2 = Dense(3)(dense2)
dense2 = Dense(1)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])

middle1 = Dense(40)(merge1)
middle1 = Dense(3)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(5)(middle1)
output = Dense(1)(middle1)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit([x1,x2], y, epochs=5000, callbacks=[early_stopping])

print(x1_predict)
print(x2_predict)

#4. 평가, 예측
y_predict = model.predict([x1_predict, x2_predict])
print(y_predict)

loss = model.evaluate([x1,x2], y, batch_size=1)
print("loss : ", loss)

# 하이퍼파라미터 튜닝
# epochs=150 input1노드=3,10,4,3,1 input2노드=3,100,4,3,1 middle노드=40,3,8,5,1
# [[50.824394]]

# epochs=60 input1노드=3,100,7,5 input2노드=3,100,8,6 middle노드=4,25,50,1
# [[37.708687]]

# epochs=60 input1노드=3,8,7,5 input2노드=3,10,8,6 middle노드=4,25,50,1
# [[37.863636]]
