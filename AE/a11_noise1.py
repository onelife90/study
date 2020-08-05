# 데이터에 노이즈를 제거해보자
#1) 입력하는 x데이터에 노이즈 추가하기 위해 정규분포 np.random.normal() 메서드 사용
#2) 원본 x데이터의 값의 범위(0~1)를 맞추기 위해 np.clip() 메서드 사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

#1. 데이터
train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2])/255

#1-2. 데이터 노이즈 추가
# 정규분포 np.random.normal(0,0.5) = 평균이 0이고, 표준편차가 0.1인 랜덤값을 만들어서 x_train에 점을 찍어줘라
# 표준편차가 커질수록 높이는 낮고 양옆 길이가 길어지는 정규분포가 생성
x_train_noise = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noise = x_test + np.random.normal(0,0.1,size=x_test.shape)

# 여기서 문제점이 있다. x데이터가 0~1 사이인데 np.random.normal로 인해 1을 더하거나 음수값이 더해지면 x데이터가 0~1 사이의 범위를 벗어남
# np.clip(배열, 최소값, 최대값)==최소값에 못 미치는 데이터들은 다 0으로 치환, 최대값 1 넘는 데이터들은 다 1로 치환
x_train_noise = np.clip(x_train_noise,a_min=0,a_max=1)
x_test_noise = np.clip(x_test_noise,a_min=0,a_max=1)

#2. 모델 구성
# 함수로 autoencoder 모델 구성
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])   # loss: 0.01
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])    # loss: 0.06

model.fit(x_train_noise, x_train, epochs=50)

#4. 예측
output = model.predict(x_test_noise)

#5. 시각화
# 이미지 다섯개를 무작위로 고름
random_img = random.sample(range(output.shape[0]), 5)

# fig = 전체 액자에 대한 변수, axes = 액자 내 여러 개의 액자에 대한 리스트
# figure 객체가 1개, axes가 10개로 구성된 plt
fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5, figsize=(20,7))

'''
a = ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))
print(f'type(a): {type(a)}')        # type(a): <class 'tuple'>
print(f'type(ax1): {type(ax1)}')    # type(ax1): <class 'matplotlib.axes._subplots.AxesSubplot'>
print(ax1)                          # AxesSubplot(0.125,0.53;0.133621x0.35)
'''

# 원본(노이즈X) 이미지
# i=인덱스, ax=axes의 자료형을 열거하는 for문 (0,ax1), (1,ax2),...(4,ax5)
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    # random_img.shape=(784, )이기 때문에 (28,28)로 reshape
    ax.imshow(x_test[random_img[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 포함 이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    # random_img.shape=(784, )이기 때문에 (28,28)로 reshape
    ax.imshow(x_test_noise[random_img[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈가 포함된 이미지를 오토 인코더가 출력한 이미지
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    # random_img.shape=(784, )이기 때문에 (28,28)로 reshape
    ax.imshow(output[random_img[i]].reshape(28,28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
# loss: 0.0019 - acc: 0.0167
