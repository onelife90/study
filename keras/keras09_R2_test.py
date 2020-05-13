# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기 / 편차=오차*2
# 레이어는 인풋과 아웃풋을 포함 5개 이상, 노드는 레이어당 각각 5개 이상
# batch_size = 1
# epochs = 100 이상
# 데이터 유지!

#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) 

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1)) 
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(2))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
model.fit(x_train, y_train, epochs=100, batch_size=1) 

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

#epochs=100, Dense=3,2,1,1
#결과값
#RMSE :  1.1952(1) / 0.5099(2)
#R2 :  0.2856(1) / 0.8699(2)

#epochs=100, Dense=5,4,2,1
#결과값
#RMSE :  1.1616(1) / 1.1750(2) / 1.2144(3) / 1.2794(4) / 0.2066(5)
#R2 :  0.3252(1) / 0.3095 (2) / 0.2625(3) / 0.9999(4) / 0.9786(5)

#epochs=100, Dense=3,1,2,3,1
#결과값
#RMSE :  1.2069(1) / 0.8834(2) / 0.4257(3)
#R2 :  0.2716(1) / 0.6097(2) / 0.9093(3)




