# 앙상블 : 음악 합주 / 어마한 데이터 셋 2개 이상을 합쳐보자

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(301,401)])

y1 = np.array([range(711,811), range(611,711)])
y2 = np.array([range(101,201), range(411,511)])

###################################
###### 여기서 부터 수정하세요. ######
###################################
x = np.transpose(x)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

# print(x.shape)
# print(y1.shape)
# print(y2.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x,y1,y2, shuffle=False, train_size=0.8
)       

# from sklearn.model_selection import train_test_split
# x2_train, x2_test, y1_train, y1_test = train_test_split(
#     x2,y1, shuffle=False, train_size=0.8
# )     

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # 함수형 모델은 input, output을 명시해줘야함

input1 = Input(shape=(2, ))
dense1_1 = Dense(80, activation='relu', name='bit1')(input1)
dense1_2 = Dense(500, activation='relu', name='bit2')(dense1_1)
dense1_3 = Dense(100, activation='relu', name='bit3')(dense1_2)
dense1_4 = Dense(200, activation='relu', name='bit4')(dense1_3)

##### output 모델 구성 #####

output1 = Dense(50)(dense1_4)
output1_2 = Dense(250)(output1)
output1_3 = Dense(400)(output1_2)
output1_4 = Dense(30)(output1_3)
output1_5 = Dense(100)(output1_4)
output1_6 = Dense(2)(output1_5)

output2 = Dense(20)(dense1_4)
output2_2 = Dense(400)(output2)
output2_3 = Dense(400)(output2_2)
output2_4 = Dense(30)(output2_3)
output2_5 = Dense(100)(output2_4)
output2_6 = Dense(2)(output2_5)

model = Model(inputs=input1, outputs=[output1_6, output2_6])
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시

model.summary()
# M1-M3가 번갈아 가면서 훈련될 예정
# model.summary()의 layer 이름 변경하는 파라미터? ==> name 파라미터

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, [y1_train, y2_train], epochs=80, batch_size=1,
          validation_split=0.3, verbose=1)
          # list로 묶어서 한번에 model.fit 완성
                   
#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test], batch_size=1)
# 전체 loss값(1)
# y1_output1에 대한 loss(1)
# y1_output1에 대한 mse(1)
# y2_output1에 대한 loss(1)
# y2_output1에 대한 mse(1)
# 총 5개의 반환값

print("loss : ", loss)

y1_predict, y2_predict = model.predict(x_test)
#(20,3)짜리 3개 왜? train_size=0.8이기 때문에
print(y1_predict)
print("==============")
print(y2_predict)

# RMSE 구하기

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
# sklearn에서는 list를 감당할 수 없다
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)

#R2 구하기
from sklearn.metrics import r2_score
R2_1 = r2_score(y1_test, y1_predict) 
R2_2 = r2_score(y2_test, y2_predict) 
print("R2_1 : ", R2_1)
print("R2_2 : ", R2_2)

# 하이퍼파라미터튜닝
# epochs=80, input노드=2,80,500,100,200 output1노드=50,250,400,30,100,2 output2노드=20,400,400,30,100,2
# RMSE1 :  14.255225487086447
# RMSE2 :  9.245407773996222
# R2_1 :  -5.111622667298617
# R2_2 :  -1.5707538318036025

# epochs=300, input노드=2,10,5,8,20 output1노드=25,35,40,32,22,2 output2노드=38,18,64,32,10,2
# RMSE1 :  7.6119855040604225
# RMSE2 :  27.437356416094566
# R2_1 :  -0.7426262650834885
# R2_2 :  -21.640857958009192

# epochs=300, input노드=2,10,5,8,20 output1노드=10,8,6,4,2,2 output2노드=9,7,5,3,1,2
# RMSE1 :  21.986380606923984
# RMSE2 :  30.277665906341518
# R2_1 :  -13.53837390052701
# R2_2 :  -26.571039180031146

# epochs=500, input노드=2,10,5,8,20 output1노드=10,8,6,4,2,2 output2노드=9,7,5,3,1,2
# RMSE1 :  14.27049282778736
# RMSE2 :  36.7775601234793
# R2_1 :  -5.12472076835881
# R2_2 :  -39.679366274770956

# epochs=80, input노드=2,10,5,8,20 output1노드=100,80,60,40,20,2 output2노드=90,70,50,30,10,2
# RMSE1 :  78.29212651232719
# RMSE2 :  31.98706729111653
# R2_1 :  -183.3505886863833
# R2_2 :  -29.77210447778698

# epochs=300, input노드=2,10,5,8,20 output1노드=100,80,60,40,20,2 output2노드=90,70,50,30,10,2
# RMSE1 :  3.1756694258173215
# RMSE2 :  1.7651674821423167
# R2_1 :  0.696695449561777
# R2_2 :  0.9062912409018753
