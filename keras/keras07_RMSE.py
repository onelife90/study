# 회귀 모델 : 수치를 넣었을 때 수치로 답한다
# 회귀 모델에서 지표가 acc라면, 엉뚱한 값에도 결과가 부여 ex) 강아지(0), 고양이(1), 공룡(0.xxx)
# MSE -> RMSE
# (실제값-예측값)^2 / n 한 후 루트 씌우기 => 루트 씌우기 때문에 값이 작을수록 좋음
# 회귀 모델이 잘 트레이닝 되었는지 판단하는 기준으로 사용
# 회귀 모델에서 가장 중요한 이슈는 오차(=실제값-예측값)가 가장 최소화되는 가장 적합한 '예측선'을 찾는 것

#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 라인 복사 : ctrl+c / 라인 삭제 : shift+del
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) # pred = predict

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1)) # activation에도 디폴트가 있다
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(350))
model.add(Dense(450))
model.add(Dense(300))
model.add(Dense(250))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # metrics에는 대괄호 필수 문법임. 훈련 돌릴 때 보여지는 부분
model.fit(x_train, y_train, epochs=80, batch_size=1) # w 가중치가 나옴

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
# model.fit에 나온 가중치로 mse를 산출 
# 그래서 y_test_predict(예측값) 나옴

print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred) #y_pred로 반환한다
# print("y_predict : \n", y_pred)

y_predict = model.predict(x_test) # 입력값에 y_test는 안되나요? ★절대안됨! x = input, y = output 이기 때문에!
print(y_predict)

from sklearn.metrics import mean_squared_error
# 사이킷런 = 케라스, 텐서플로우 이전의 킹왕짱
#사이킷런에는 RMSE가 없어서 mse를 불러옴 -> 루트 씌울거임

def RMSE(y_test, y_predict): # def = 함수를 호출하겠다 () = 입력값
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 반환 = 출력값
            #sqrt = 루트
print("RMSE : ", RMSE(y_test, y_predict))

# R2 = 1에 가까울수록 좋은 값. accuracy와 유사한 결과를 도출
# mse, RMSE는 값이 작을수록 좋은 값
