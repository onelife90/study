# 과제 2
# Sequential형으로 완성하시오
# 하단에 주석으로 acc와 loss 결과 명시
# CNN=4차원 / 현재 x데이터(3차원)==> 4차원 reshape ==> input_shape=3차원 

import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
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

x_train = x_train.reshape(-1,28,28,1).astype('float32')/255
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255
# print(x_train.shape)        # (60000, 28, 28, 1)
# print(x_test.shape)         # (10000, 28, 28, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(900, (3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(810, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(10))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=300, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.3, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=300)

print("loss: ", loss)
print("acc: ", acc)

# 튜닝
# epochs=100, batch=300, 노드=10,900,max3,drop0.3,flat
