import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
# mnist 손글씨로 된 7만장의 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# 0~255의 숫자가 컬럼으로 찍혀있다
# print("y_train: ", y_train[0])

# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])        # 랜덤색깔
# print(x_train[0].shape)       # (28,28)
# plt.imshow(가로, 세로)==가로, 세로를 넣어주면 이미지를 출력
# plt.show()

# 0~9까지(손글씨 숫자) 10개로 분류
# 분류모델로 쓰려면 one-hot 인코딩을 사용해서 2차원으로 변환

# 데이터 전처리 1. one-hot 인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255
# reshape로 4차원을 만든 다음 실수형으로 타입을 바꾸고 255로 나눈다(x의 데이터가 255이기 때문에)
# minmax와 비슷한 정규화 과정. 255로 나누면 최대값=1, 최소값=0이 되기 때문에

# x의 데이터 양이 6만개라서 x의 데이터 범위를 0~1로 정규화
# x_train = x_train / 255
# 이 방식도 가능. x_train의 최대값이 255이기 때문에 범위 최대값=1, 최소값=0

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(900, (3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
# Dropout 전까지의 모든 레이어의 40%를 제거

model.add(Conv2D(50, (3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(15, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(10))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=600)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)

# 하이퍼파라미터튜닝
#loss:  nan
#acc:  0.09799999743700027

# Conv2D(10),Max(pool=2),Conv2D(80),
#loss:  nan
#acc:  0.09799999743700027


# Conv2D(10, ker_s=2),Max(pool=2),Conv2D(900, ker_s=3),max(pool=2),Drop=0.4,Conv2D(50, ker_s=3),max(pool=2),Drop=0.2,Conv2D(15,ker_s=2),max,Flatten
#loss:  
#acc:  
