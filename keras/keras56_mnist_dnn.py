import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D
from keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)        # (60000, 10)
print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255

print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=(28*28)))
model.add(Dense(30))
model.add(Dropout(0.3))
model.add(Dense(800))
model.add(Dropout(0.4))
model.add(Dense(90))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
earlystopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=600, validation_split=0.3, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=600)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)

# 하이퍼파라미터 튜닝
# epochs=100, batch=600, 노드=10,300,Drop(0.3),8000,Drop(0.4),900,Drop(0.2)
