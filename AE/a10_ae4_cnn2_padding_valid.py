# cnn으로 오토인코더 구성하시오
# padding='valid'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

#1. 데이터
train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, x_train.shape[1],x_train.shape[2],1)/255
x_test = x_test.reshape(-1, x_test.shape[1],x_test.shape[2],1)/255

#2. 모델 구성
# 함수로 autoencoder 모델 구성
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3), padding='valid', input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
    model.add(Conv2D(filters=77, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(1, kernel_size=(3,3), padding='same', activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

# model.summary()

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['acc'])   # loss: 0.01
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])    # loss: 0.06

model.fit(x_train, x_train, epochs=1)

#4. 예측
output = model.predict(x_test)

#5. 시각화
# 이미지 다섯개를 무작위로 고름
random_img = random.sample(range(output.shape[0]), 5)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5, figsize=(20,7))

# x_test가 출력된 이미지를 위에 그린다
# i=인덱스, ax=axes의 자료형을 열거하는 for문 (0,ax1), (1,ax2),...(4,ax5)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    # random_img.shape=(28,28,1 )이기 때문에 (28,28)로 reshape
    ax.imshow(x_test[random_img[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    # random_img.shape=(28,28,1 )이기 때문에 (28,28)로 reshape
    ax.imshow(output[random_img[i]].reshape(28,28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# loss: 0.1621 - acc: 0.8034
