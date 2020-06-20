# 최적화 함수
# 가중치 갱신은 오차 함수를 각 가중치로 미분한 값을 바탕으로 갱신해야 할 방향과 어느 정도로 갱신할지 결정
# 미분에 의해 구한 값을 학습 속도, epochs, 과거의 가중치 갱신량 등을 근거로 어떻게 가중치 갱신에 반영할지 정함

# 학습률
# 각 층의 가중치를 한 번에 어느 정도로 변경할지 결정하는 하이퍼파라미터

# 네트워크 구조(은닉층 수, 은닉층 유닛 수)는 자유롭게 설정
# 은닉층이 많아지면 입력층에 가까운 가중치를 적절하게 갱신하기 어렵고 학습이 진행되지 않음
# 중요성이 낮은 특징량을 추출해버려서 과학습하기 쉬워질 가능성

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
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#3. 컴파일, 훈련
def funA():
    global lr
    lr = 0.01

def funB():
    global lr
    lr = 0.1

def funC():
    global lr
    lr = 1.0

# funA()
# funB()
funC()

sgd = optimizers.SGD(lr=lr)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train,y_train, batch_size=32, epochs=3, verbose=1, validation_data=(x_test,y_test))

#4. 평가, 예측
score = model.evaluate(x_test, y_test)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
# funA
# evaluate loss: 1.3652963027954101      
# evaluate acc: 0.7329999804496765  

# funB
# evaluate loss: 0.7262102193832397      
# evaluate acc: 0.7770000100135803 

# funC
# evaluate loss: 1.7941453876495361      
# evaluate acc: 0.33399999141693115  
