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

# print(x.shape)    # (100,2)
# print(y1.shape)   # (100,2)
# print(y2.shape)   # (100,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x,y1,y2, shuffle=False, train_size=0.8)       

# print(x_train.shape)    # (80, 2)
# print(y1_test.shape)    # (20, 2)
# print(y2_test.shape)    # (20, 2)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # 함수형 모델은 input, output을 명시해줘야함

input1 = Input(shape=(2, ))
# input(shape=(2, )) 2콤마!!
# shape=(x, ) 나중에 인풋이 많아지면 헷갈리니 shape와 dim 구분
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense1 = Dense(500, activation='relu', name='bit2')(dense1)
dense1 = Dense(8, activation='relu', name='bit3')(dense1)
dense1 = Dense(10, activation='relu', name='bit4')(dense1)
dense1 = Dense(500, activation='relu', name='bit5')(dense1)
dense1 = Dense(8, activation='relu', name='bit6')(dense1)
dense1 = Dense(20, activation='relu', name='bit7')(dense1)

##### output 모델 구성 #####
output1 = Dense(15)(dense1)
output1 = Dense(45)(output1)
output1 = Dense(35)(output1)
output1 = Dense(55)(output1)
output1 = Dense(25)(output1)
output1 = Dense(2)(output1)

output2 = Dense(35)(dense1)
output2 = Dense(45)(output2)
output2 = Dense(15)(output2)
output2 = Dense(30)(output2)
output2 = Dense(25)(output2)
output2 = Dense(2)(output2)

model = Model(inputs=input1, outputs=[output1, output2])
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시
# model.summary()의 layer 이름 변경하는 파라미터? ==> name 파라미터

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, [y1_train, y2_train], epochs=300, batch_size=1, validation_split=0.3, verbose=3)
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
print("RMSE : ", (RMSE1+RMSE2)/2)

#R2 구하기
from sklearn.metrics import r2_score
R2_1 = r2_score(y1_test, y1_predict) 
R2_2 = r2_score(y2_test, y2_predict) 
print("R2_1 : ", R2_1)
print("R2_2 : ", R2_2)
print("R2 : ", (R2_1+R2_2)/2)

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
# RMSE1 :  3.17 / 3.21
# RMSE2 :  1.76 / 3.43
# R2_1 :  0.69 / 0.68
# R2_2 :  0.90 / 0.64

# epochs=5000, input노드=2,10,5,8,20 output1노드=100,80,60,40,20,2 output2노드=90,70,50,30,10,2
# RMSE1 :  152.17972531486768
# RMSE2 :  57.554382440387585
# R2_1 :  -695.501317200258
# R2_2 :  -98.62426881486904

# epochs=300, input노드=2,10,500,8,20 output1노드=100,80,60,400,20,2 output2노드=90,70,50,300,10,2
# RMSE1 :  7.318102411903175
# RMSE2 :  50.09376288323822
# R2_1 :  -0.6106653507098667
# R2_2 :  -74.4702279639728

# epochs=1000, input노드=2,10,500,8,20 output1노드=100,80,60,400,20,2 output2노드=90,70,50,300,10,2
# RMSE1 :  1.1391739350232433
# RMSE2 :  30.38211806200208
# R2_1 :  0.9609709096470274
# R2_2 :  -26.761596930328814

# epochs=300, input노드=2,10,500,8,10,500,8,20 output1노드=100,80,60,400,20,2 output2노드=90,70,50,300,10,2
# RMSE1 :  17.673095908493874
# RMSE2 :  34.99508145554693
# R2_1 :  -8.393633653859334
# R2_2 :  -35.83175116031171

# epochs=300, input노드=2,10,500,8,10,500,8,20 output1노드=15,45,35,55,25,2 output2노드=35,45,15,30,25,2
# RMSE1 :  9.271965203369358
# RMSE2 :  37.89167425208559
# R2_1 :  -1.5855440220298402
# R2_2 :  -42.18132263537343

# epochs=300, input노드=2,10,500,8,10,500,8,20 output1노드=15,450,35,55,25,2 output2노드=35,45,150,300,25,2
# RMSE1 :  41.55549887941749
# RMSE2 :  14.587416632616355
# RMSE :  28.071457756016923
# R2_1 :  -50.93562367269979
# R2_2 :  -5.399781173339317
# R2 :  -28.167702423019556

# epochs=3000, input노드=2,10,500,8,10,500,8,20 output1노드=15,450,35,55,25,2 output2노드=35,45,150,300,25,2
# RMSE1 :  1.5638237284208671
# RMSE2 :  1.1749756451777804
# RMSE :  1.3693996867993237
# R2_1 :  0.9264497848549731
# R2_2 :  0.9584791649094453
# R2 :  0.9424644748822092
