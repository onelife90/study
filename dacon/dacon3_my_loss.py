import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, Lambda, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#1. csv 불러오기
train = pd.read_csv('./data/dacon/comp3/train_features.csv', header=0, index_col=0)
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', header=0, index_col=0)
submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv', header=0, index_col=0)

#1-1. shape 확인
# id당 375행의 데이터
print(train.shape)          # (1050000, 5)==(2800*375, 5)
print(train_target.shape)   # (2800, 4)
print(test.shape)           # (262500, 5)==(700*375, 5)
print(submit.shape)         # (700, 4)

#1-2. 결측치 확인
# print(train.isnull().sum())         # 결측치 X
# print(train_target.isnull().sum())  # 결측치 X
# print(test.isnull().sum())          # 결측치 X

#1-3. 시계열 확인
# print(train.head())           # 일정한 시간 간격으로 가속도 측정

#1-4. 넘파이 저장
x = train.values
x_pred = test.values
y = train_target.values
np.save('./data/dacon/comp3/train_features.npy', arr=x)
np.save('./data/dacon/comp3/test_features.npy', arr=x_pred)
np.save('./data/dacon/comp3/train_target.npy', arr=y)

#1-6. Scaler 후 reshape
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_pred = scaler.fit_transform(x_pred)

x = x.reshape(2800,375,5,1)
x_pred = x_pred.reshape(700,375,5,1)

#1-7. train_test_split
x_train, x_test, y_train, y_test = tts(x,y, random_state=88, test_size=0.2)
# print(x_train.shape)    # (2240, 375, 5, 1)
# print(y_test)

#2-1. my_loss 구성
weight1 = np.array([1,1,0,0])
weight2 = np.array([0,0,1,1])
# 리스트*정수 계산 가능
# print(weight1*-3)

def my_loss(y_test, y_pred):
    # Lambda 레이어는 인공적인 코드를 포함해서 인공지능이나 설계 프로그램에서 쓰는 Lisp 언어에서 물려받음
    # 함수를 딱 한줄로만 만들어주는 훌륭한 녀석
    div_res = Lambda(lambda x: x[0]/x[1])([(y_pred-y_test),(y_test+0.000001)])
    # 즉, (y_pred-y_test) / (y_test+0.000001)를 실행하겠다
    # 왜 y_test+0.000001 인가? 0으로 나누어지는 걸 방지하기 위해
    return K.mean(K.square(div_res))
    # Mean of a tensor, alongside the specified axis.
    # square : Element-wise square.(제곱)

def my_loss_E1(y_test, y_pred):
    return K.mean(K.square(y_test-y_pred)*weight1)/2e+04
    # (실제-예측값)*weight1의 제곱 / 2e+04(==2*10의 4제곱)

def my_loss_E2(y_test, y_pred):
    div_res = Lambda(lambda x: x[0]/x[1])([(y_pred-y_test),(y_test+0.000001)])
    return K.mean((K.square(div_res)*weight2))

#2-2. 모델 구성 및 컴파일
def set_model(train_target):
    activation='elu'
    padding='same'
    nf=16
    fs=(3,1)

    input1 = Input(shape=(375,5,1))
    dense1 = BatchNormalization()(input1)
    dense1 = MaxPooling2D(pool_size=(2,1))(dense1)
    dense1 = Conv2D(nf*75, fs, padding=padding, activation=activation)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D(pool_size=(2,1))(dense1)
    dense1 = Conv2D(nf*25, fs, padding=padding, activation=activation)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D(pool_size=(2,1))(dense1)
    dense1 = Conv2D(nf*5, fs, padding=padding, activation=activation)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D(pool_size=(2,1))(dense1)
    # dense1 = Conv2D(nf*16, fs, padding=padding, activation=activation)(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = MaxPooling2D(pool_size=(2,1))(dense1)
    # dense1 = Conv2D(nf*32, fs, padding=padding, activation=activation)(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = MaxPooling2D(pool_size=(2,1))(dense1)
    dense1 = Flatten()(dense1)
    dense1 = Dense(128, activation='elu')(dense1)
    dense1 = Dense(64, activation='elu')(dense1)
    dense1 = Dense(32, activation='elu')(dense1)
    dense1 = Dense(16, activation='elu')(dense1)
    output1 = Dense(4, activation='elu')(dense1)

    model = Model(inputs=input1, outputs=output1)

    # 전역변수 global
    global weight2
    if train_target==1:
        weight2 = np.array([0,0,1,0])
    else:
        weight2 = np.array([0,0,0,1])
    
    if train_target==0:
        model.compile(loss=my_loss_E1, optimizer='adam')
    else:
        model.compile(loss=my_loss_E2, optimizer='adam')

    return model

#3. 훈련
def train(model, x, y):
    e_stop = EarlyStopping(monitor='val_loss', patience=4)
    hist = model.fit(x,y, epochs=80, batch_size=80, validation_split=0.2, callbacks=[])
    fig, loss_ax = plt.subplots()
    # twinx() 공통적인 x축을 갖지만 서로 다른 y축을 사용할 경우
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epochs')
    loss_ax.set_ylabel('loss')
    loss_ax.legend()
    # plt.show()

    return model

for train_target in range(3):
    model = set_model(train_target)
    train(model, x_train, y_train)
    y_pred = model.predict(x_pred)

    if train_target==0:
        submit.iloc[:,1] = y_pred[:,0]
        submit.iloc[:,2] = y_pred[:,1]
    elif train_target==1:
        submit.iloc[:,3] = y_pred[:,2]
    elif train_target==2:
        submit.iloc[:,3] = y_pred[:,3]

#4. 평가, 예측
mse = model.evaluate(x_test, y_test, batch_size=10)
y_pred = model.predict(x_pred)
print("y_pred : \n", y_pred)
print("mse: ", mse)

#5. submit할 csv 파일 생성
y_pred = pd.DataFrame(y_pred, index=np.arange(2800,3500))
y_pred.to_csv('./data/dacon/comp3/submit_my_loss_.%5f.csv'%(mse), header=["X","Y","M","V"], index=True, index_label="id")

# y_pred :
#  [[1.5517836  0.29417682 2.239648   0.4538685 ]
#  [1.290546   0.51877916 2.3048644  0.508647  ]
#  [0.70265025 0.6500039  1.9277257  0.4049458 ]
#  ...
#  [0.64869547 0.47757912 1.5759571  0.31412333]
#  [1.3691134  0.38725084 2.9310992  0.5756059 ]
#  [0.5757394  0.38771105 1.6395435  0.31495303]]
# mse:  0.00034130964390247494
