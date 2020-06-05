#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array(range(711,811))

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)  # (100,3)
print(y.shape)  # (100,1)

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
model.add(Dense(9000))
model.add(Dense(1))
# y 데이터가 1개 칼럼이기 때문에 output_Dense가 1

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_split=0.3, verbose=3)
#verbose=0 --> 결과값만 빠르게 스캔
#verbose=1 --> 통상 보던 디폴트
#verbose=2 --> 진행 bar가 없어짐
#verbose=3 --> epochs 진행, 예측값, RMSE, R2 결과
           
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
