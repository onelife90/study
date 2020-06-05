# 과제 train_test_split 함수를 1번만 써서 model.fit에서 validation_data를 다른 파라미터로 대체해보시오
# 1) train_test_split 함수 2개 중 x_val, y_val이 포함된 두번째 함수 삭제
# 2) model.fit validation_data=(x_val, y_val)에 x_val, y_val를 삭제했기 때문에 쓸수가없음. 다른 파라미터로 대체
# 3) 그럼 어떤 파라미터? 구글링에 소질이 없나봄.. 꿀팁! model.fit validation_까지만 치면 연관검색어에 model.fit validation_data 바로 밑에 split이 나옴
# 4) model.fit validation_split를 쓰면 train 비율에 validation 비율을 split 하는 것임
# 이게 가능한 이유는 train에 validation 검증이 포함되어 있기 때문임

#1. 데이터
import numpy as np
# x_train = np.array([1,2,3,4,...,100]) #엄청나게 비효율적
x = np.array(range(1,101)) # 1부터 시작해서 101-1까지 나열
y = np.array(range(101,201)) # y=wx+b / w=1, b=100이 되는 함수

from sklearn.model_selection import train_test_split
# train_test_split라는 함수가 사이킷런에 구현되어있음
# Q. x_train, x_test, x_val, y_train, y_test, y_val이라는 변수를 선언하지 않았는데요?
# A. train_test_split이라는 함수에 이미 다 포함되어 있는 변수란다
# train set과 test set을 손쉽게 분리할 수 있음
# 하지만 과적합의 우려가 있기 때문에 validation set을 추가하여 방지
# Train+Validation 훈련하면서 검증, TV로 생각하자
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.6)       
print(x_train)
# print(x_val)
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
model.fit(x_train, y_train, epochs=300, batch_size=1,validation_split=0.3)
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
