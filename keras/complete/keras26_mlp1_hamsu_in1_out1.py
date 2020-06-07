# keras14_mlp.py를 Sequential에서 함수형으로 변경
# earlyStopping 적용

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array([range(101,201), range(711,811), range(100)])

x = np.transpose(x)
y = np.transpose(y)
# print(x.shape)    # (100,3)
# print(y.shape)    # (100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=99, train_size=0.6)       
# print(x_train.shape)    # (60, 3)
# print(x_test.shape)     # (40, 3)
# print(y_train.shape)    # (60, 3)
# print(y_test.shape)     # (40, 3)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))
dense1 = Dense(1000, activation='relu')(input1)
dense1 = Dense(10)(dense1)
dense1 = Dense(5)(dense1)
dense1 = Dense(1000)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(5)(dense1)

output1 = Dense(100)(dense1)
output1 = Dense(80)(output1)
output1 = Dense(60)(output1)
output1 = Dense(40)(output1)
output1 = Dense(20)(output1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.3, verbose=3, callbacks=[early_stopping])
           
#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
print("RMSE : ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print("R2 : ", r2)

# epochs=300, input1노드=3,1000,10,5,1000,10,5 output노드=100,80,60,40,20,10,3
#RMSE :  23.95
#R2 :  0.34

# epochs=300, input1노드=3,1000,10,5000,400,5 output노드=100,80,60,40,20,10,3
#RMSE :  17.58
#R2 :  0.64

# epochs=20, input1노드=3,1000,10,5000,400,5 output노드=100,80,60,40,20,10,3
#RMSE :  10.94
#R2 :  0.86

# epochs=20, input1노드=3,1000,10,50,400,5 output노드=100,80,60,40,20,10,3
#RMSE :  11.99
#R2 :  0.83

# epochs=30, input1노드=3,1000,10,50,400,5 output노드=100,80,600,40,200,10,3
#RMSE :  26.09
#R2 :  0.22

# epochs=30, input1노드=3,1000,10,50,400,5 output노드=100,40,500,10,3
#RMSE :  1.70
#R2 :  0.996

# epochs=30, input1노드=3,1000,10,50,400,5 output노드=40,500,10,3
#RMSE :  1.43
#R2 :  0.997

# epochs=20, input1노드=3,1000,10,50,400,5 output노드=40,5000,10,3
#RMSE :  5.42
#R2 :  0.96

# epochs=20, input1노드=3,5,10,9000,3 output노드=40,5000,6,3
#RMSE : 0.0013 
#R2 :  0.999999998

# epochs=30, input1노드=3,5,10,8000,3 output노드=40,5000,6,3
#RMSE : 0.004 
#R2 :  0.99999997
