# 반복학습
# 모델의 정확도를 높이기 위해서 동일한 훈련 데이터를 사용하여 여러 번 학습
# 반복 학습. 학습할 횟수는 epochs
# 무조건 epochs를 높인다고 해서 모델의 정확도가 계속 오르는 것은 아님

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
model.add(Activation('softmax'))

#3. 컴파일, 훈련
sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

def funA():
    global epochs
    epochs = 5

def funB():
    global epochs
    epochs = 10

def funC():
    global epochs
    epochs = 60

# funA()
funB()
# funC()

hist = model.fit(x_train,y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

#4. 평가, 예측
score = model.evaluate(x_test, y_test)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
# funA
# evaluate loss: 0.6566096363067627      
# evaluate acc: 0.8090000152587891    

# funB
# evaluate loss: 0.6282063140869141      
# evaluate acc: 0.7850000262260437 

# funC
# evaluate loss: 0.6465612735748291      
# evaluate acc: 0.7730000019073486   

#5. 시각화
plt.plot(hist.history['acc'], label='acc', ls='-', marker='o')
plt.plot(hist.history['val_acc'], label='val_acc', ls='-', marker='x')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()
