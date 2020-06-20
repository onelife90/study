# 모델 평가
# 훈련 데이터로 학습을 수행하여 모델의 튜닝이 성공적 진행
# 모델이 훈련 데이터에만 통용되도록 학습해버린 가능성(과학습)도 존재하므로 모델의 성능을 제대로 평가 X
# 학습에 이용하지 않았던 데스트 데이터를 사용해서 모델로 분류하고, 모델의 평가를 실시
# score = model.evaluate(x_test, y_test, verbose=1)
# x_test, y_test는 평가용 입력 데이터와 지도 데이터
# evaluate() 메서드는 손실 함수의 값과 정확도를 얻을 수 있으며 위 예제의 경우 모두 score에 저장

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
