# 모델 학습
# 모델에 훈련 데이터를 전달하여 학습 실시
# model.fit(x_train, y_train, verbose=1, epochs=3)
# x_tarin, y_tarin은 각각 학습용 입력 데이터와 지도 데이터
# verbose에 지정한 숫자로 학습의 진척 상황 표시 조정
# verbose=1 학습 진척 출력 verbose=0 학습 진척 미출력
# epochs는 동일한 데이터 셋으로 몇 번 반복 학습할지 지정
# fit()메서드는 학습용 데이터를 순서대로 모델에 입력하고 출력 및 지도 데이터 간의 차이가 작아지도록 각 뉴런의 가중치 갱신
# 이에 따라 오차가 감소하고 모델의 예측 정확도가 향상

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

#4. 시각화
plt.plot(hist.history['acc'], label='acc', ls='-', marker='o')
plt.plot(hist.history['val_acc'], label='val_acc', ls='-', marker='x')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()
