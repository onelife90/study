# 데이터 두개 이상을 사용해보자
# ex) data = 삼성, 하이닉스 주가 output = 다우지수, xx지수 (output은 2개 이상 나올 수 있음)


#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array([range(101,201), range(711,811), range(100)])
# 파이썬에는 list가 있는데 덩어리를 []로 묶어줘야함
# 1st 덩어리 : w=1, b=100
# 2nd 덩어리 : w=1, b=400
# 3rd 덩어리 : w=1

x = np.transpose(x)
y = np.transpose(y)
# 처음엔 np.transpose(x)만 코딩하고 왜 안되지 함
# 안되는 이유? transpose한 값을 x라는 공간에 저장을 하지 않았기 때문! 즉, 변수의 초기화 단계는 필수
print(x.shape)
print(y.shape)

# (3,100) 3행 100열 / 뭔가 이상? 3행이면 가로 3줄
# 통상적인 엑셀 기입방법을 떠올려보자 : 데이터를 추가할 때(↓)는 행에 추가가 됨 
# x(컬럼) = 날씨, 용돈, 삼성 / y = sk
# 외워라~ 열우선, 행무시
# input_dim=3, 3개 컬럼을 사용하겠다 / 다른 x 데이터 종류 +1 추가하려면 에러가 뜸. 왜? input_dim=3이기 때문에
# (3,100) 3행 100열 이기 때문에 우리가 통상적으로 생각하는 엑셀로 바꿔줘야함

'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, train_size=0.6
)       

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
model.add(Dense(3))
# 역시나 output_Dense도 3으로 변경 

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=300, batch_size=1,
            validation_split=0.3)
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
'''
