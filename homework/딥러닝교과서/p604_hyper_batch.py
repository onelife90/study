# 미니배치 학습
# 모델의 학습을 실시할 때 한 번에 모델에 전달하는 입력 데이터 수를 바꿀 수 있음
# 한 번에 전달하는 데이터 수를 배치크기 batch_size
# 각 데이터의 손실과 손실 함수의 기울기(가중치를 어떻게 갱신할 것인가)를 구하지만 가중치 갱신은 구해진 기울기의 평균으로 한 번만 실시
# 편향된 데이터가 많을 때는 배치 크기를 크게 하고 유사한 데이터가 많을 때는 배치 크기를 작게 하는 조정
# 배치 크기를 1로 하는 방식 : 온라인 학습(확률적 경사하강법)
# 배치 크기를 전체 데이터 수로 지정하는 방식 : 배치 학습(경사하강법)
# 이들의 중간이 되는 방식 : 미니 배치 학습

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
sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

def funA():
    global batch_size
    batch_size = 16

def funB():
    global batch_size
    batch_size = 32

def funC():
    global batch_size
    batch_size = 64

# funA()
# funB()
funC()

hist = model.fit(x_train,y_train, batch_size=batch_size, epochs=3, verbose=1)

#4. 평가, 예측
score = model.evaluate(x_test, y_test)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
# funA
# evaluate loss: 0.9579451732635498      
# evaluate acc: 0.6790000200271606   

# funB
# evaluate loss: 0.7276678609848023      
# evaluate acc: 0.7950000166893005 

# funC
# evaluate loss: 0.7384610652923584      
# evaluate acc: 0.8230000138282776
