from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
# 전체 데이터 중 학습에 300장, 테스트에 100장의 데이터 사용
# Conv 층은 4차원 배열(배치크기X가로X세로X채널)
# mnist 데이터는 RGB가 아니라 원래 3차원 데이터이므로 미리 4차원으로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:300].reshape(-1,28,28,1)
x_test = x_test[:100].reshape(-1,28,28,1)
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:100]

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
model.add(Activation='relu')
model.add(Conv2D(64, (3,3)))
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=1, batch_size=128,
          validation_data=(x_test, y_test))

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print('loss: ', loss)
print('acc: ', acc)


