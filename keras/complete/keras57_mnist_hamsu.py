import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000, )
# print(y_test.shape)         # (10000, )

# #1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)
# print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1,28*28).astype('float32')/255
x_test = x_test.reshape(-1,28*28).astype('float32')/255
# print(x_train.shape)        # (60000, 784)      # 왜 784가 되나? reshape하기 전에 차원의 구성요소를 곱한 값은 항상 같아야하므로!
# print(x_test.shape)         # (10000, 784)

#2. 모델 구성
input1 = Input(shape=(784,))
dense1 = Dense(28)(input1)
dense1 = Dense(56)(dense1)
dense1 = Dense(112)(dense1)
dense1 = Dense(224)(dense1)
dense1 = Dense(448)(dense1)
dense1 = Dense(784)(dense1)
dense1 = Dropout(0.3)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(10)(dense1)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=600, validation_split=0.3, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)
print("loss: ", loss)
print("acc: ", acc)

# 튜닝
# epochs=74, batch=100, 노드=5000,Drop0.1,7030,Drop0.1,10
# loss:  0.31035348621197045
# acc:  0.9175999760627747
