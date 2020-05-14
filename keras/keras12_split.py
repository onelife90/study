# 지금껏 데이터를 가내수공업으로 만들어서 사용
# 사이킷런에서 자동으로 데이터 골라 훈련해주는 train_test_split
# ex) 차를 왜타죠? 걸어가도 끝까지 가는데. 더 빠르고 효율적이기 때문에 사용
# train(초1-1), val(초1-2), test(초2-1)
# 훈련이 처음부터 안되면 train(초1-1) -> val(초1-2) : 다음 단계로 검증이 안된다
# 그렇기 때문에 train(초1-1)60%, val(초1-2)20%, test(초2-1)20%를 랜덤으로 데이터를 섞어 훈련시켜주는 것이 train_test_split

#1. 데이터
import numpy as np
# x_train = np.array([1,2,3,4,...,100]) #엄청나게 비효율적
x = np.array(range(1,101)) # 1부터 시작해서 101-1까지 나열
y = np.array(range(101,201)) # y=wx+b / w=1, b=100이 되는 함수

from sklearn.model_selection import train_test_split
#train_test_split라는 함수가 사이킷런에 구현되어있음
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, 
    train_size=0.6
)       
# random_state=66 난수 지정하고 연속으로 실행해도 똑같은 값이 나온다
# train_size=0.6 전체 데이터 셋의 60%를 차지. test_size 60% + train_size 40%
x_val, x_test, y_val, y_test = train_test_split(
# test 대신 train이 와도 무방    
    # x_test, y_test, random_state=99,
    x_test, y_test,
    test_size=0.5
)       

print(x_train)
print(x_val)
print(x_test)

'''
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

#하이퍼파라미터튜닝
#epochs=500, 히든레이어=9, 노드=(5,3000,50,10,5)x2,1
#RMSE : 6.72(1) / (2) / (3)
#R2 : 0.95(1) / (2) / (3)
