# 100개의 데이터를 훈련시켜보자 -> 배열 사용
# 총 데이터를 6:2:2로 나눠보자(데이터 전처리)

#1. 데이터
import numpy as np
# x_train = np.array([1,2,3,4,...,100]) #엄청나게 비효율적
x = np.array(range(1,101)) # 1부터 시작해서 101-1까지 나열
y = np.array(range(101,201)) # y=wx+b / w=1, b=100이 되는 함수

#데이터 전처리
x_train = x[:60] # 처음 인덱스(1)부터 60번째 인덱스(60)까지
x_val = x[60:80]
x_test = x[80:] # 끝: == 명시하지 않으면 끝까지

y_train = x[:60] 
y_val = x[60:80]
y_test = x[80:] 

print(x_train)
print(x_val)
print(x_test)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1)) 
model.add(Dense(5000))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5000))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=50, batch_size=1,
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
#epochs=50, 히든레이어=10, 노드=(5,5000,50,10,5)x2,1
#RMSE :  0.5750(1) / 0.0614(2) / 0.4346(3)
#R2 :  0.9900(1) / 0.9998(2) / 0.9943(3)

#epochs=100, 히든레이어=10, 노드=(5,5000,50,10,5)x2,1
#RMSE :  0.0023(1) / 0.2461(2) / 0.071(3) / 0.0861(4) / 0.0592(5)
#R2 :  0.9999998(1) / 0.9981(2) / 0.9998(3) / 0.9997(4) / 0.9998(5)

#epochs=100, 히든레이어=10, 노드=(5,3000,30,10,5)x2,1
#RMSE :  0.0001(1) / 7.27(2)
#R2 :  0.9999999995(1) / -0.58(2)

#epochs=50, 히든레이어=10, 노드=(5,3000,30,10,5)x2,1
#RMSE :  0.0030(1) / 0.4721(2) / 0.2222(3)
#R2 :  0.9999997(1) / 0.9932(2) / 0.9985(3)

#epochs=80, 히든레이어=10, 노드=(5,3000,30,10,5)x2,1
#RMSE :  46(1) / 6.1911(2) / 0.4946(3)
#R2 :  -62(1) / 0.9999999998(2) / 0.9926(3)

#epochs=100, 히든레이어=10, 노드=(5,2000,20,10,5)x2,1
#RMSE :  0.0436(1) / 22(2) / 0.0060(3) / 0.0003(4) / 1.86e-05(5) / 0.0038(6)
#R2 :  0.99994(1) / -14(2) / 0.999998(3) / 0.9999999996(4) / 0.99999999998(5) / 0.99999995(6)
#값이 왔다리 갔다리..별로 좋은 모델은 아닌듯

#epochs=80, 히든레이어=10, 노드=(5,2000,20,10,5)x2,1
#RMSE :  30(1)
#R2 :  -26(1)

#epochs=500, 히든레이어=10, 노드=(5,2000,20,10,5)x2,1
#RMSE : 1.4675(1) / 0.00080(2) / 0.0003(3)
#R2 : 0.999999999999(1) / 0.99999998(2) / 0.9999999995(3)

#epochs=300, 히든레이어=10, 노드=(5,2000,20,10,5)x2,1
#RMSE : 0.0005(1) / 1.23e-05(2) / 0.050(3)
#R2 : 0.999999992(1) / 0.9999999999995(2) / 0.99999992(3)
#모델을 바꿔보자

#epochs=300, 히든레이어=9, 노드=(5,1000,10,5)x2,1
#RMSE : 0.0004(1) / 0.0055(2) / 2.51(3)
#R2 : 0.999999993(1) / 0.99999990(2) / 0.803)
