# 데이터 두개 이상을 사용해보자
# ex) data = 삼성, 하이닉스 주가 output = 다우지수, xx지수 (output은 2개 이상 나올 수 있음)
# MLP(Multi Layer Perceptron) 다층신경망. Perceptron? 시각과 뇌의 기능을 모델화한 학습 기계

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array([range(711,811)])
# 통상적으로 y는 하나의 값이 좋다
# input_data = 삼성, 하이닉스 / output_data = 삼성 // 이렇게 하나만 나오는 결과가 더 높은 신뢰성
# 그래서 output_data를 한 컬럼으로 체체체인지

# (3,100) 3행 100열 / 뭔가 이상? 3행이면 가로 3줄
# 통상적인 엑셀 기입방법을 떠올려보자 : 데이터를 추가할 때(↓)는 행에 추가가 됨 
# x(컬럼) = 날씨, 용돈, 삼성 / y = sk
# 외워라~ 열우선, 행무시
# input_dim=3, 3개 컬럼을 사용하겠다 / 다른 x 데이터 종류 +1 추가하려면 에러가 뜸. 왜? input_dim=3이기 때문에
# (3,100) 3행 100열 이기 때문에 우리가 통상적으로 생각하는 엑셀로 바꿔줘야함
x = np.transpose(x)
y = np.transpose(y)
# 처음엔 np.transpose(x)만 코딩하고 왜 안되지 함
# 안되는 이유? transpose한 값을 x라는 공간에 저장을 하지 않았기 때문! 즉, 변수의 초기화 단계는 필수
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.6)       
# print(x_train)
# print(x_val)
# print(x_test)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 3))
# 여지껏 input_dim=1이었지만, 데이터 컬럼이 3개 이므로, input_dim=3으로 변경
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
# y 데이터가 1개 칼럼이기 때문에 output_Dense가 1

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.3)
# validation_split=0.3 / train_size=60 / 60% X 30% = 18%
# 즉, train=42%, val=18%의 비율로 훈련됨
           
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
#epochs=500, 히든레이어=8, 노드=5,1000,10,10,5,1000,10,5,1
#RMSE : 0.0036(1) / 0.0001(2) / 0.0001(3)
#R2 : 0.99999998(1) / 0.99999999998(2) / 0.99999999997(3)

#epochs=300, 히든레이어=8, 노드=5,1000,10,10,5,1000,10,5,1
#RMSE :  8.79e-05(1) / 0.0079(2) / 0.0009(3)
#R2 :  0.999999999990568(1) / 0.99999992(2) / 0.999999998(3)

#epochs=200, 히든레이어=8, 노드=5,1000,10,10,5,1000,10,5,1
#RMSE :  0.0215(1) / 0.1597(2)
#R2 :  0.9999992(1) / 0.99996(2)

#epochs=100, 히든레이어=8, 노드=5,1000,10,10,5,1000,10,5,1
#RMSE : 0.1785(1) / 0.2546(2) / 0.0071(3)
#R2 : 0.99996(1) / 0.99993(2) / 0.99999993(3)

#epochs=500, 히든레이어=7, 노드=5,1000,10,5,1000,10,5,1
#RMSE : 4.13(1)
#R2 : 0.97(1)

#epochs=300, 히든레이어=7, 노드=5,1000,10,5,1000,10,5,1
#RMSE : 0.0031(1) / 0.0205(2) / 20.6(3)
#R2 : 0.99999998(1) / 0.9999995(2) / 0.52(3)

#epochs=500, 히든레이어=7, 노드=5,(3000,10,5)x2,1
#RMSE : 0.2678(1) / 17.31(2)
#R2 : 0.99990(1) / 0.65(2)

#epochs=300, 히든레이어=7, 노드=5,(3000,10,5)x2,1
#RMSE : 9.27(1)
#R2 : 0.88(1)

#epochs=500, 히든레이어=7, 노드=5,1000,10,9000,5,5000,10,1
#RMSE : 0.0654(1) / 0.0040(2) / 0.0145(3)
#R2 : 0.999994(1) / 0.99999998(2) / 0.9999997(3)

#epochs=300, 히든레이어=7, 노드=5,1000,10,9000,5,5000,10,1
#RMSE : 0.0019(1) / 0.0188(2) / 0.0032(3)
#R2 : 0.999999995(1) / 0.9999996(2) / 0.99999998(3)

#epochs=100, 히든레이어=7, 노드=5,1000,10,9000,5,5000,10,1
#RMSE : 7.76(1)
#R2 : 0.92(1)

#epochs=5000, 히든레이어=4, 노드=5,1000,10,9000,1
#RMSE : 0.0010(1) / (2) / (3)
#R2 : 0.999999998(1) / (2) / (3)

#epochs=500, 히든레이어=4, 노드=5,1000,10,9000,1
#RMSE : 0.0043(1) / 0.7383(2) / 0.0037(3)
#R2 : 0.999997(1) / 0.9992(2) / 0.99999998(3)
