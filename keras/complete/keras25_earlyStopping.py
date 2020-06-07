# R2의 값이 올라가다가 갑자기 흔들리는 지점이 생김
# RMSE는 값이 내려가다가 갑자기 수치가 흔들리는 지점이 생김
# 저 지점이 오기 전에 끊을 수 있을까? 최고점을 잡기는 힘들다
# 최고점을 조금 지난 상태에서 epochs를 확인하여 끊을 수 있다. 그것이 earlyStopping

#1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311,411), range(411,511)])
x2 = np.array([range(711,811), range(711,811), range(511,611)])

y1 = np.array([range(101,201), range(411,511), range(100)])

###################################
###### 여기서 부터 수정하세요. ######
###################################
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

# print(x1.shape)   # (100,3)
# print(x2.shape)   # (100,3)
# print(y1.shape)   # (100,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1,x2,y1, random_state=66, train_size=0.8)       


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # 함수형 모델은 input, output을 명시해줘야함

input1 = Input(shape=(3, ))
dense1 = Dense(80, activation='relu', name='bit1')(input1)
dense1 = Dense(500, activation='relu', name='bit2')(dense1)
dense1 = Dense(400, activation='relu', name='bit3')(dense1)
dense1 = Dense(100, activation='relu', name='bit4')(dense1)
dense1 = Dense(200, activation='relu', name='bit5')(dense1)
dense1 = Dense(20, activation='relu', name='bit6')(dense1)

input2 = Input(shape=(3, ))
dense2 = Dense(400, activation='relu', name='camp1')(input2)
dense2 = Dense(200, activation='relu', name='camp2')(dense2)
dense2 = Dense(300, activation='relu', name='camp3')(dense2)
dense2 = Dense(100, activation='relu', name='camp4')(dense2)
dense2 = Dense(50, activation='relu', name='camp5')(dense2)
dense2 = Dense(5, activation='relu', name='camp6')(dense2)

# M1, M2 두 개 모델의 레이어들을 엮어주는 API 호출
from keras.layers.merge import concatenate
# concatenate=사슬 같이 잇다
merge1 = concatenate([dense1, dense2])
# merge1 레이어를 만들어줌 (M1_끝 레이어 + M2_끝 레이어)
middle1 = Dense(100)(merge1)
middle1 = Dense(80)(middle1)
middle1 = Dense(60)(middle1)
middle1 = Dense(70)(middle1)
middle1 = Dense(90)(middle1)
middle1 = Dense(40)(middle1)
# merge 된 이후에 딥러닝이 구성
# output이 y1이므로 총 1개로 도출되어야 함

##### output 모델 구성 #####
output1 = Dense(50)(middle1) # y_M1의 가장 끝 레이어가 middle1
output1_2 = Dense(250)(output1)
output1_3 = Dense(400)(output1_2)
output1_4 = Dense(30)(output1_3)
output1_5 = Dense(100)(output1_4)
output1_6 = Dense(3)(output1_5)

model = Model(inputs=[input1, input2], outputs=output1_6)
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시

# model.summary()
# model.summary()의 layer 이름 변경하는 파라미터? ==> name 파라미터

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
# earlyStopping 파라미터를 호출
# 카멜케이스 형태로 앞의 'E'가 대문자
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
# monitor=모니터링 지표
# patience=선생님의 개그를 견디는 횟수 / mode가 min이면 그 횟수 이하, max면 그 횟수 이상
# mode=min, max, auto

model.fit([x1_train, x2_train], y1_train, epochs=500, batch_size=1, validation_split=0.25, verbose=3, callbacks=[early_stopping])
          # 콜백에는 리스트 형태                 
              
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
# y1_output1에 대한 loss(1)
# y1_output1에 대한 mse(1)
# 총 2개의 반환값

print("loss : ", loss)

y1_predict = model.predict([x1_test, x2_test])
print(y1_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
# sklearn에서는 list를 감당할 수 없다
RMSE = RMSE(y1_test, y1_predict)
print("RMSE : ", RMSE)

#R2 구하기
from sklearn.metrics import r2_score
R2 = r2_score(y1_test, y1_predict) 
print("R2 : ", R2)

# 하이퍼파라미터튜닝
# epochs=1000, input1노드=3,30,500,400,200,100,10 input2노드=3,50,500,400,200,10,5 
# middle노드= 10,20,..,100 output노드=50,200,60,200,70,200,80,200,90,3
#RMSE :  19.74
#R2 :  0.50 

# epochs=500, input1노드=3,30,500,400,200,100,10 input2노드=3,50,500,400,200,10,5 
# middle노드= 10,20,..,100 output노드=50,200,60,200,70,200,80,200,90,3
#RMSE :  27.40
#R2 :  0.04

# epochs=3000, input1노드=3,30,5000,400,200,100,10 input2노드=3,50,5000,400,200,10,5 
# middle노드= 10,20,..,500,60,70,80,90,100 output노드=500,200,60,200,70,200,80,200,90,3
#RMSE : 19.61
#R2 : 0.51

# epochs=100, input1노드=3,30,5000,400,200,100,10 input2노드=3,50,5000,400,200,10,5 
# middle노드= 10,20,..,500,60,70,80,90,100 output노드=500,200,60,200,70,200,80,200,90,3
#RMSE : 23.03
#R2 : 0.32

# epochs=100, input1노드=3,300,5000,4000,2000,1000,100 input2노드=3,500,5000,4000,2000,1000,500 
# middle노드= 100,200,...,900,100 output노드=5000,2000,6000,2000,7000,2000,8000,2000,9000,3
#RMSE : 113
#R2 : -15

# epochs=100, input1노드=3,20,50,30,60,80,10 input2노드=3,70,60,40,20,10,50 
# middle노드= 100,80,60,40,20,10,30,50,70,90 output노드=150,145,140,135,130,125,120,115,110,3
#RMSE : 32
#R2 : -0.32
