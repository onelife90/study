# 앞뒤가 똑같은 오토인코더 = x가 인풋이면 x도 아웃풋
# 대한민국 축구대표팀. 좋은 경험이었다? --> 그 과정에서 특징을 추출함
# 특징을 추출한다? 어디서 배웠지? PCA(차원축소, 압축)

# 예시) 기태 사진에 잉크를 뿌렸다. 오토인코더에 넣으면 기태 얼굴이 나옴. 하지만 흐릿하게 나옴
# 잉크가 특징이 아니고 기태 얼굴이 특징이기 때문에

# 더 깔끔하게 만들고싶다? 그래서 나온게 GAN

import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()
# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리
x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255
#print(x_train.shape)        # (60000, 784)
#print(x_test.shape)         # (10000, 784)

#2. 모델구성
input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
# x데이터가 0~255에서 정규화를 해줬기 때문에 0~1사이로 수렴하는 sigmoid를 사용
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

#3. 컴파일, 훈련
autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2, callbacks=[earlystopping])

#4. 평가, 예측
decoded_img = autoencoder.predict(x_test)

#5. 시각화
n=10
plt.figure(figsize=(20,4))

for i in range(n):
    # subplot(2,10,1),(2,10,2),...,(2,10,10) 순으로 그림
    ax = plt.subplot(2,n,i+1)
    # (28,28) 사이즈로 x_test 보기
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()

    # x축 y축 없애기
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # subplot(2,10,11),(2,10,12),...,(2,10,20) 순으로 그림
    ax = plt.subplot(2,n,i+1+n)
    # (28,28) 사이즈로 decoded_img = pred의 값 보기
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()

    # x축 y축 없애기
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
