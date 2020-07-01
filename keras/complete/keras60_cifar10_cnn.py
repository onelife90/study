# cifar10 : 10가지 이미지를 찾는 데이터
# CNN_함수형 3차원 필요. 현재 x데이터(4차원) ==> input_shape 3차원 넣기

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
# print(y_train.shape)        # (50000, 10)
# print(y_test.shape)         # (10000, 10)

#2. 모델 구성
input1 = Input(shape=(32,32,3))
dense1 = Conv2D(32, (2,2), padding='same')(input1)
dense1 = Conv2D(64, (2,2), padding='same')(dense1)
dense1 = Conv2D(128, (2,2), padding='same')(dense1)
dense1 = Conv2D(256, (2,2), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=3, padding='same')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(224, (2,1), padding='same')(dense1)
dense1 = Conv2D(160, (2,1), padding='same')(dense1)
dense1 = Conv2D(69, (2,1), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2, padding='same')(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.3, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)

# 튜닝
