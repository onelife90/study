import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)        # (60000, 10)
print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(x_train.shape[0], 392, 2).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 392, 2).astype('float32')/255

print(x_train.shape)        # (60000, 392, 2)
print(x_test.shape)         # (10000, 392, 2)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(392,2)))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.3, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)
