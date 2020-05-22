# Standardization, minmax가 제일 중요! ==> 데이터 전처리
# ★ 정규화 minmax = x-최소/최대-최소
# minmax : min과 max를 사용해서 0~1 사이의 범위로 데이터를 표준화해주는 변환
# sklearn.preprocessing.MinMaxScaler() method   # preprocessing 전처리
# sklearn.preprocessing.minmax_scale() 함수

# ★ 표준화 standardScaler x-평균/표편
# Standardization(표준화) : 서로 다른 단위가 있을 때 중심이 0이고 편차가 1인 분포로 만드는것

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [2000,3000,4000], [3000,4000,5000], [4000,5000,6000],
           [100,200,300]])
           # 10개 w=1, 3개 w=100
           # 그래프로 그리면 오른쪽으로 많이 치우친다. 괜찮은 데이터라 할 수 없음
           # 머신은 w=100에 더 신경을 쓴다 --> 0~1로 범위를 축소
           # minmax scaler / standard scaler
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])
# y는 표준화 안해도 됨. 왜? 전처리는 x값만 해주고 y는 결과값이기 때문에
x_predict = array([55,65,75])
# x_predict 전처리를 안하면 [[2464.2986]]로 너무 다른 값이 나옴.

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# MaxAbsScaler : 0을 기준으로 절대값이 가장 큰 수가 1 또는 -1이 되도록 변환
# RobustScaler : 중앙값이 0, IQR이 1이 되도록 변환
scaler = MinMaxScaler()
scaler = StandardScaler()
scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x)                   # 실행
x = scaler.transform(x)         # 변환. 항상 선실행 후변환
x_predict = scaler.transform(x_predict)           
# 앞서 x에게 선실행 후변환 했기 때문에 x_predict도 변환이 됨
print(x)


print("x.shape : ", x.shape)        # (14,3)
print("y.shape : ", y.shape)        # (14,)
print("x_predict.shape : ", x_predict.shape)       # (3,)

x = x.reshape(x.shape[0], x.shape[1], 1)   
x_predict = x_predict.reshape(1,3,1)

print(x.shape)          #(13,3,1)
print(x_predict.shape)  #(1,3,1)

#2. 모델구성

input1 = Input(shape=(3,1))
dense1 = LSTM(10, return_sequences=True)(input1)
dense1 = LSTM(10)(dense1)
dense1 = Dense(5)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x,y, epochs=5000, callbacks=[early_stopping])

print(x_predict)

#4. 평가, 예측
y_predict = model.predict(x_predict)
print(y_predict)

loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
