# cifar10 : 10가지 이미지를 찾는 데이터
# DNN_함수형 Dense 2차원 필요. 현재 x데이터(4차원) ==> 2차원으로 reshape

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print("x_train[0]: \n", x_train[0])
# print("y_train[0]: \n", y_train)

# print(x_train.shape)        # (50000, 32, 32, 3)
# print(x_test.shape)         # (10000, 32, 32, 3)
# print(y_train.shape)        # (50000, 1)
# print(y_test.shape)         # (10000, 1)

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (50000, 10)
print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1,32*32*3).astype('float32')/255
x_test = x_test.reshape(-1,32*32*3).astype('float32')/255
# print(x_train.shape)        # (50000, 3072)
# print(x_test.shape)         # (10000, 3072)

#2. 모델 구성
input1 = Input(shape=(3072, ))
dense1 = Dense(32)(input1)
dense1 = Dense(64)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(224)(dense1)
dense1 = Dense(160)(dense1)
dense1 = Dense(69)(dense1)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)

#튜닝
#epochs=81, batch_size=100, 노드=100,9000(softmax),70,drop0.1,40,20
# loss:  1.8786926889419555
# acc:  0.4575999975204468
