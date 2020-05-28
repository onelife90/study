import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28,28,1).astype('float32')/255

#2. 모델구성
model = Sequential()
model.add(Dense(100, (2,2), padding='same', input_shape=(784,1)))
model.add(Dense(300))
model.add(Dropout(0.3))
model.add(Dense(8000))
model.add(Dropout(0.5))
model.add(Dense(90))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=600)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)
