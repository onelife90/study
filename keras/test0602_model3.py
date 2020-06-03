# 앙상블. 함수형 모델로 구성
# LSTM+LSTM 구현

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

#2-1. 데이터 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle=True)
hite = np.load('./data/hite.npy', allow_pickle=True)
# 저장하는 데이터는 파이썬 피클을 사용. 하지만, 로딩시 임의의 코드를 불러올 수 있음
# 임의의 코드말고 우리가 저장한 데이터를 불러와야 하므로 allow_pickle을 사용
# print(samsung.shape)        # (509, 1)
# print(hite.shape)           # (509, 5)

#2-1. 데이터 자르기
# split 함수를 쓰기 전에 samsung.shape=(509,1)
# 이 상태 그대로 split를 쓴다면 (504,6,1)이 된다. 
# 왜? 509개의 스칼라가 쭉 이어진 벡터 1개가 split 함수를 거치면 []+[]=[[ ]] 이런 형태로 변하기 때문

#2-1-1. 데이터 벡터화(1컬럼짜리 데이터)
samsung = samsung.reshape(samsung.shape[0], )

#2-1-2. 2차원으로 바꿔주는 split 함수
size = 6
def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)
dataset = split_x(samsung, size)
samsung = split_x(samsung, size)
# print(samsung.shape)        # (504, 6)
# print(samsung)

#2-1-3. samsung 데이터 x, y 자르기
x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]
# print(x_sam.shape)         # (504, 5)
# print(y_sam.shape)         # (504, )

#2-1-4. 3차원 reshape
x_sam = x_sam.reshape(-1,5,1)
# print(x_sam.shape)         # (504, 5, 1)
# print(y_sam.shape)         # (504, )

#2-1-5. hite 데이터 행맞추기
x_hite = hite[5:510, :] 

#2-1-6. hite 데이터 3차원 reshape
x_hite = x_hite.reshape(-1,5,1)
# print(x_hite.shape)     # (504, 5, 1)

#2-2. 모델 구성
input1 = Input(shape=(5,1))
x1 = LSTM(10)(input1)
x1 = Dense(700)(x1)
x1 = Dense(500)(x1)
x1 = Dense(300)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(5,1))
x2 = LSTM(100)(input2)
x2 = Dense(20)(x2)
x2 = Dense(700)(x2)
x2 = Dense(500)(x2)
x2 = Dense(300)(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(100)(x2)

merge = Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x_sam, x_hite], y_sam, epochs=100)
