# 함수형 모델
# 함수는 재사용을 위해 생성
# A_sequential 모델 + B_sequential 모델. 어떻게 한번에 묶을까?
# A_sequential 모델==A 함수, B_sequential 모델==B함수로 묶어서 전체를 또다른 하나의 신경망으로 표현

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array(range(711,811))

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, train_size=0.6
)       

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # 함수형 모델은 input, output을 명시해줘야함
# model = Sequential()
# model.add(Dense(5, input_dim = 3))
# model.add(Dense(4))
# model.add(Dense(1))

input1 = Input(shape=(3, )) #행무시 열우선. 100행 3열이기에 3열 표시
# 변수명은 소문자로 암묵적인 룰
# 함수형 모델에서는 keras.layer의 계층 친구인 Input을 명시해줘야함
dense1 = Dense(5000, activation='relu')(input1)
# activation=활성화 함수 # 앞단의 아웃풋이 뒤 꽁지에 붙음
dense2 = Dense(4000, activation='relu')(dense1)
output1 = Dense(1)(dense2) #activation에도 디폴트가 있음

model = Model(inputs=input1, outputs=output1)
# 순차적 모델은 model = Sequential()이라고 명시를 하고 시작했지만,
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시해줘야 함. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs=100, batch_size=1,
            validation_split=0.3, verbose=3)
                   
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

# 하이퍼파라미터튜닝
# epochs=100, 히든레이어=2, 노드=3,5000,4000,1 validation_split=0.3
#RMSE : 0.4290 
#R2 : 0.9997 

# epochs=50, 히든레이어=5, 노드=3,500,400,300,200,100,1 validation_split=0.3
#RMSE : 2.0403
#R2 : 0.994

# epochs=100, 히든레이어=5, 노드=3,1000,800,400,200,100,1 validation_split=0.3
#RMSE : 0.6540
#R2 : 0.9994

# epochs=300, 히든레이어=5, 노드=3,1000,800,400,200,100,1 validation_split=0.3
#RMSE : 27.40
#R2 : -0.04

# epochs=100, 히든레이어=10, 노드=3,1000,900,800,..,100,1 validation_split=0.3
#RMSE : 1.57
#R2 : 0.997

# epochs=100, 히든레이어=10, 노드=3,1000,900,1000,700,1000,500,1000,300,1000,100,1 validation_split=0.3
#RMSE : 1.20
#R2 : 0.998
