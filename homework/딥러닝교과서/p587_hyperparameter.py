# 하이퍼파라미터 : 네트워크를 구성할 때 사람이 조정해야 하는 파라미터가 존재

import numpy as np
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],784)[:6000]
x_test = x_test.reshape(x_test.shape[0],784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=784))
# 하이퍼파라미터 : 활성화함수
model.add(Activation("sigmoid"))
# 하이퍼파라미터 : 은닉층 수, 은닉층의 채널 수
model.add(Dense(128))
model.add(Activation("sigmoid"))
# 하이퍼파라미터 : 드롭아웃 비율(rate)
model.add(Dropout(rate=0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

#3. 컴파일, 훈련
# 하이퍼파라미터 : 학습률(lr)
sgd = optimizers.SGD(lr=0.1)
# 하이퍼파라미터 : 최적화함수(optimizer)
# 하이퍼파라미터 : 오차함수(loss)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
# metrics는 평가 함수이므로 학습 자체와는 관계 X
# 하이퍼파라미터 : batch_size
# 하이퍼파라미터 : epochs
hist = model.fit(x_train,y_train, batch_size=50, epochs=35, verbose=1, validation_data=(x_test,y_test))

#4. 평가, 예측
score = model.evaluate(x_test, y_test)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
