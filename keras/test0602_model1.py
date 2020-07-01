# 앙상블. 함수형 모델로 구성
# Dense + Dense

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

#2-1. 데이터 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle=True)
hite = np.load('./data/hite.npy', allow_pickle=True)
# 저장하는 데이터는 파이썬 피클을 사용. 하지만, 로딩시 임의의 코드를 불러올 수 있음
# 임의의 코드말고 우리가 저장한 데이터를 불러와야 하므로 allow_pickle을 사용
# print(samsung.shape)        # (509, 1)
# print(hite.shape)           # (509, 5)

# split 함수를 쓰기 전에 samsung.shape=(509,1)
# 이 상태 그대로 split를 쓴다면 (504,6,1)이 된다. 
# 왜? 509개의 스칼라가 쭉 이어진 벡터 1개가 split 함수를 거치면 []+[]=[[ ]] 이런 형태로 변하기 때문

#2-1-1. 데이터 벡터화(1컬럼짜리 데이터)
samsung = samsung.reshape(samsung.shape[0], )
# print(samsung.shape)        # (509,)

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
# 0:5 컬럼까지 x로 두고, 마지막 컬럼을 y로 쓰겠다
x1 = samsung[:, 0:5]
y = samsung[:, 5]
# print(x_sam.shape)      # (504, 5)
# print(y_sam.shape)      # (504,)

# print(samsung.shape)    # (504, 6)
# print(hite.shape)       # (509, 5)
# 이 상태에서 모델 구성을 하면 동작이 가능할까? 모델 구성은 가능하지만 fit에서 error가 남
# ValueError: All input arrays (x) should have the same number of samples.
# 앙상블 모델에서는 행을 맞춰야함!
# 그럼 어떻게 자를까? hite의 가장 오래된 행을 잘라보는 편이 낫다

#2-1-4. hite 데이터 행맞추기
x2 = hite[5:510, :] 
# print(x_hite.shape)     # (504, 5)

#2-1-5. train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.6, random_state=99)
# 변수명_train, 변수명_test라고 명시해야 에러 X
# print(x1_train.shape)       # (302, 5)
# print(x1_test.shape)        # (202, 5)
# print(x2_train.shape)       # (302, 5)
# print(x2_test.shape)        # (202, 5)

#2-2. 모델 구성
input1 = Input(shape=(5, ))
x1 = Dense(100)(input1)
x1 = Dense(1000)(x1)
x1 = Dense(3000)(x1)
x1 = Dense(5000)(x1)

input2 = Input(shape=(5, ))
x2 = Dense(100)(input2)
x2 = Dense(1000)(x2)
x2 = Dense(3000)(x2)
x2 = Dense(5000)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stop = EarlyStopping(monitor='loss', patience=3, mode='auto')
modelpath = './model/test0602/{epoch:02d}-{val_loss:.4f}.hdf5'
check_p = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False)
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stop, check_p])

#4. 평가, 예측
loss, mse = model.evaluate([x1_test, x2_test], y_test)

# print(x1_test[-1, :])       # [49700 49000 48500 47600 47200] 벡터
x1_pred = x1_test[-1, :].reshape(-1,5)
print(x1_pred)              # [[49700 49000 48500 47600 47200]] 행렬
# 벡터를 2차원으로 reshape ==> 행렬
x2_pred = x2_test[-1, :].reshape(-1,5)
print(x2_pred)              # [[19900 20150 19800 20100 254623]] 행렬
# 벡터를 2차원으로 reshape ==> 행렬
y_pred = model.predict([x1_pred, x2_pred])
print("y_pred: \n", y_pred)
