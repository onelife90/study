from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

#1. 데이터
train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2])/255

#2. 모델 구성
# 함수로 autoencoder 모델 구성
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['acc'])   # loss: 0.01
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])    # loss: 0.06

model.fit(x_train, x_train, epochs=50)

#4. 예측
output = model.predict(x_test)

#5. 시각화
# 이미지 다섯개를 무작위로 고름
random_img = random.sample(range(output.shape[0]), 5)

# fig = 전체 액자에 대한 변수, axes = 액자 내 여러 개의 액자에 대한 리스트
# figure 객체가 1개, axes가 10개로 구성된 plt
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5, figsize=(20,7))

'''
a = ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))
print(f'type(a): {type(a)}')        # type(a): <class 'tuple'>
print(f'type(ax1): {type(ax1)}')    # type(ax1): <class 'matplotlib.axes._subplots.AxesSubplot'>
print(ax1)                          # AxesSubplot(0.125,0.53;0.133621x0.35)
'''

# x_test가 출력된 이미지를 위에 그린다
# i=인덱스, ax=axes의 자료형을 열거하는 for문 (0,ax1), (1,ax2),...(4,ax5)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    # random_img.shape=(784, )이기 때문에 (28,28)로 reshape
    ax.imshow(x_test[random_img[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    # random_img.shape=(784, )이기 때문에 (28,28)로 reshape
    ax.imshow(output[random_img[i]].reshape(28,28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
