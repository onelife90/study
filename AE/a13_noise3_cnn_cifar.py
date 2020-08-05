# 컬러 이미지인 cifar10으로 오토인코더 구성해보자
# 컬러이기 때문에 노이즈를 굳이 추가해줄 필요가 없음

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import random

#1. 데이터
train_set, test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

# print(f'x_train.shape: {x_train.shape}')    # x_train.shape: (50000, 32, 32, 3)   
# print(f'x_test.shape: {x_test.shape}')      # x_test.shape: (10000, 32, 32, 3)  

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, x_train.shape[1],x_train.shape[2],x_train.shape[3])/255
x_test = x_test.reshape(-1, x_test.shape[1],x_test.shape[2],x_train.shape[3])/255

#2. 모델 구성
# 함수로 autoencoder 모델 구성
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(3, kernel_size=(2,2), padding='same', activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=128)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])   # loss: 0.01
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])    # loss: 0.06

model.fit(x_train, x_train, epochs=10)

#4. 예측
output = model.predict(x_test)

#5. 시각화
# 이미지 다섯개를 무작위로 고름
random_img = random.sample(range(output.shape[0]), 5)

# fig = 전체 액자에 대한 변수, axes = 액자 내 여러 개의 액자에 대한 리스트
# figure 객체가 1개, axes가 10개로 구성된 plt
fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5, figsize=(20,7))

'''
a = ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))
print(f'type(a): {type(a)}')        # type(a): <class 'tuple'>
print(f'type(ax1): {type(ax1)}')    # type(ax1): <class 'matplotlib.axes._subplots.AxesSubplot'>
print(ax1)                          # AxesSubplot(0.125,0.53;0.133621x0.35)
'''

# 원본 이미지
# i=인덱스, ax=axes의 자료형을 열거하는 for문 (0,ax1), (1,ax2),...(4,ax5)
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_img[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_img[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
# loss: 0.0011 - acc: 0.8816
