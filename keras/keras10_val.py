#validation 추가
#train만 하면 교과서만 푼 것
#train, val은 훈련하면서 검증 model.fit에서 실행이 되고 교과서 + 모의고사를 같이 푼 것
#다시말해, 훈련시키고 컨닝(1epoch), 훈련시키고 컨닝(2epoch), ... 반복적인 행위가 됨
#test는 model.evaluate에서 실행

#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16,17,18]) 
x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])
# w 구할 수 있나? 앞으로는 유추할 수 없는 w값이 생성이 됨

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1)) 
model.add(Dense(50))
model.add(Dense(5000))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5000))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=80, batch_size=1,
            validation_data=(x_val, y_val)) 
            #val 값은 train과 같이 훈련되어야 하기에 model.fit에 포함

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("mse : ", mse)

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

#하이퍼파라미터튜닝
#epochs=50, 히든레이어=10, 노드=5,50,5000,50,5,10,5000,50,5,10,1
#RMSE :  0.0022(1) / 0.0483(2)
#R2 :  0.9999997(1) / 0.9988(2)

#epochs=50, 히든레이어=9, 노드=5,5000,50,10,...
#RMSE :  6.46e-05(1) / 0.004(2) / 0.0054(3)/ 7.74e-05(4) / 1.20e-06(5)
#R2 :  0.9999999997(1) / 0.9999998(2) / 0.99998(3) / 0.999999996(4) / 0.9999999999992(5)
