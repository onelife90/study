# Sequential형으로 완성하시오
# 하단에 주석으로 acc와 loss 결과 명시
# DNN_2차원 필요. 현재 x데이터(3차원)==> 2차원 reshape ==> input_shape=1차원 벡터

import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)
# print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1,28*28).astype('float32')/255
x_test = x_test.reshape(-1,28*28).astype('float32')/255
print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(784, )))
model.add(Dense(9000))
model.add(Dropout(0.3))
model.add(Dense(500))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(10))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=300, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.3, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=300)

print("loss: ", loss)
print("acc: ", acc)
