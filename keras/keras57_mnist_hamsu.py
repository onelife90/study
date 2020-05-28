import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout

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
print(x_train.shape)        # (60000, 784)      # 왜 784가 되나? reshape하기 전에 차원의 구성요소를 곱한 값은 항상 같아야하므로!
print(x_test.shape)         # (10000, 784)

#2. 모델 구성
input1 = Input(shape=(784,))
dense1 = Dense(500)(input1)
dense1 = Dense(1000)(dense1)
dense1 = Dense(5000)(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(7000)(dense1)
dense1 = Dropout(0.3)(dense1)
dense1 = Dense(900)(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(100)(dense1)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=300)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=600)
print("loss: ", loss)
print("acc: ", acc)

# epochs=100, batch=600, 노드=5000,Drop0.4,70,Drop0.3,30,drop0.1,10
#loss:  0.3085723387263715
#acc:  0.9169999957084656
