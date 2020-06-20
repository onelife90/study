# 드롭아웃 : 과학습 방지하여 모델의 정확도를 높이는 방법 중 하나
# 유닛의 일부가 학습할 때마다 무작위로 제거(정확히 설명하면 0으로 덮어쓰기)
# 따라서 신경망은 특정 뉴런의 존재에 의존할 수 없게 되어 보다 범용적인 특징을 학습
# model.add(Dropour(0.5))
# rate는 제거할 유닛의 비율. 드롭아웃을 사용하는 위치, 인수 rate는 모두 하이퍼파라미터

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
