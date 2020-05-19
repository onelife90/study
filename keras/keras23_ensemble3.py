# 앙상블 : 음악 합주 / 어마한 데이터 셋 2개 이상을 합쳐보자

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

# print(x1.shape)
# print(x2.shape)
# print(y1.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1,x2,y1, shuffle=False, train_size=0.8
)       

# from sklearn.model_selection import train_test_split
# x2_train, x2_test, y1_train, y1_test = train_test_split(
#     x2,y1, shuffle=False, train_size=0.8
# )     

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # 함수형 모델은 input, output을 명시해줘야함

input1 = Input(shape=(3, ))
dense1_1 = Dense(80, activation='relu', name='bit1')(input1)
dense1_2 = Dense(500, activation='relu', name='bit2')(dense1_1)
dense1_3 = Dense(100, activation='relu', name='bit3')(dense1_2)
dense1_4 = Dense(200, activation='relu', name='bit4')(dense1_3)

input2 = Input(shape=(3, ))
dense2_1 = Dense(400, activation='relu', name='camp1')(input2)
dense2_2 = Dense(200, activation='relu', name='camp2')(dense2_1)
dense2_3 = Dense(100, activation='relu', name='camp3')(dense2_2)
dense2_4 = Dense(50, activation='relu', name='camp4')(dense2_3)

# M1, M2 두 개 모델의 레이어들을 엮어주는 API 호출
from keras.layers.merge import concatenate
# concatenate=사슬 같이 잇다
merge1 = concatenate([dense1_2, dense2_2])
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

model.summary()
# M1-M3가 번갈아 가면서 훈련될 예정
# model.summary()의 layer 이름 변경하는 파라미터? ==> name 파라미터

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit([x1_train, x2_train],
          y1_train, epochs=80, batch_size=1,
          validation_split=0.3, verbose=1)
          # list로 묶어서 한번에 model.fit 완성
                   
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                           y1_test, batch_size=1)
# y1_output1에 대한 loss(1)
# y1_output1에 대한 mse(1)
# 총 2개의 반환값

print("loss : ", loss)

y1_predict = model.predict([x1_test, x2_test])
#(20,3)짜리 3개 왜? train_size=0.8이기 때문에
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
# epochs=3000, input1노드=3,80,500,100,200 input2노드=400,200,100,50 
# middle노드= 100,80,60,70,90,40 output1노드=50,250,400,30,100,3
#RMSE :  36.2384159416326
#R2 :  -38.49542225439916

# epochs=1000, input1노드=3,80,500,100,200 input2노드=400,200,100,50 
# middle노드= 100,80,60,70,90,40 output1노드=50,250,400,30,100,3
#RMSE :  0.28 / 0.56 / 0.40
#R2 :  0.9998 / 0.9996 / 0.9997

# epochs=1000, input1노드=3,80,500,400,100,200 input2노드=400,200,300,100,50,5 
# middle노드= 100,80,60,70,90,40 output1노드=50,250,400,30,100,3
#RMSE : 8.02/ /
#R2 : 0.91/  / 
