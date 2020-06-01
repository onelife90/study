# cifar10 : 10가지 이미지를 찾는 데이터
# LSTM_함수형 2차원 필요. 현재 x데이터(4차원) ==> input_shape 2차원 넣기

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

x_train = x_train.reshape(-1,8,32*3*4).astype('float32')/255
x_test = x_test.reshape(-1,8,32*3*4).astype('float32')/255
# print(x_train.shape)        # (50000, 8, 384)
# print(x_test.shape)         # (10000, 8, 384)

#2. 모델 구성
input1 = Input(shape=(8,384))
dense1 = LSTM(50)(input1)
dense1 = Dense(5000)(dense1)
dense1 = Dense(4000)(dense1)
dense1 = Dense(1000)(dense1)
dense1 = Dropout(0.1)(dense1)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False)
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, callbacks=[earlystopping, checkpoint])

model.save('./model/sample/cifar10/cifar10_checkpoint_best.h5')


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)
