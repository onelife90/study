# 100개의 데이터를 훈련시켜보자 -> 배열 사용
# 총 데이터를 6:2:2로 나눠보자(데이터 전처리)

#1. 데이터
import numpy as np
# x_train = np.array([1,2,3,4,...,100]) #엄청나게 비효율적
x = np.array(range(1,101)) # 1부터 시작해서 101-1까지 나열
y = np.array(range(101,201)) # y=wx+b / w=1, b=100이 되는 함수

#데이터 전처리
x_train = x[:60] # 처음 0번째부터 인덱스(1) ~ 60번째 인덱스(61) 전까지
x_val = x[60:80] # 60번째 인덱스(61) ~ 80번째 인덱스(81) 전까지
x_test = x[80:] # 80번째 인덱스(81) ~ 끝까지

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
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=300, batch_size=1,
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
