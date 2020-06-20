# 모델에 의한 분류
# x_test의 첫 사진 1장의 숫자를 예측
# pred = np.argmax(model.predict(x_test[0]))
# print("예측치 :" + str(pred))

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
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

#3. 컴파일, 훈련
sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train,y_train, batch_size=50, epochs=35, verbose=1, validation_data=(x_test,y_test))

#4. 평가, 예측
score = model.evaluate(x_test, y_test)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
# evaluate loss: 0.4777046117782593      
# evaluate acc: 0.8370000123977661  

# 테스트 데이터의 첫 10장을 표시
for i in range(10):
    plt.subplot(1,10,1+1)
    plt.imshow(x_test[i].reshape((28,28)), "gray")
plt.show()

# x_test의 첫 10장의 예측된 라벨 표시
pred = np.argmax(model.predict(x_test[:10]), axis=1)
print(pred)
# [7 6 1 0 4 1 4 9 6 9]

# argmax()는 배열의 최대 요소의 인덱스 반환하는 함수
# predict() 메서드로 0~9까지의 숫자 배열이 출력되며 argmax() 함수로 출력된 배열의 최대 요소를 돌려줌으로써 예측된 숫자가 어디에 가장 가까운지 보기 쉽게
