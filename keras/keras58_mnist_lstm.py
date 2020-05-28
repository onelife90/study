import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)        # (60000, 10)
print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2],1).astype('float32')/255
x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2],1).astype('float32')/255

print(x_train.shape)        # (60000, 784, 1)
print(x_test.shape)         # (10000, 784, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(784,1)))
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=600)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=600)

print("loss: ", loss)
print("acc: ", acc)

# 하이퍼파라미터튜닝
# epochs=100, batch=0, 노드=LSTM10,500,Drop(0.5),50,Drop(0.2)
#loss: 
#acc: 
