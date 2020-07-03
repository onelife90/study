# 112번 파일이 과적합되었다
# 과적합 피하기 3. BatchNormalization
# w*x+b를 구하는 계산 과정에서 값을 0~1 사이로 수렴시키겠다
# BatchNormalization 사용 시에는 별도로 Activation을 import
# Activation 이전에 BatchNormalization을 해야함

# 활성화 함수의 역사
#1) 계단 함수(무조건 0 or 1)
#2) sigmoid(중간 손실을 방지)
#3) relu (음수 값 손실 되는 문제)
#4) leakyrelu(음수 값이 계속 내려감)
#5) elu(음수 상계에 대해 제한을 걸자)
#6) selu 

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)        # (50000, 32, 32, 3)
# print(x_test.shape)         # (10000, 32, 32, 3)
# print(y_train.shape)        # (50000, 1)
# print(y_test.shape)         # (10000, 1)

#1-1. 데이터 전처리
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (50000, 100)
# print(y_test.shape)         # (10000, 100)

#2. 모델구성
model = Sequential()

model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.3)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
# evaluate하면 loss, acc가 출력
print("loss: ", loss)
print("acc: ", acc)

#5. 시각화
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2,1,2)
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()
plt.show()

# loss:  12.381278596496582
# acc:  0.10000000149011612
