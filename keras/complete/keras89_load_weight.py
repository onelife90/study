import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
# mnist 손글씨로 된 7만장의 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 1. one-hot 인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

#2. 모델구성
model = Sequential()
model.add(Conv2D(28, (3,2), input_shape=(28,28,1)))
model.add(Conv2D(56, (3,2), padding='same'))
model.add(Conv2D(112, (3,2), padding='same'))
model.add(Conv2D(168, (3,2), padding='same'))
model.add(Conv2D(224, (3,2), padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=3))
model.add(Conv2D(196, (3,1), padding='same'))
model.add(Conv2D(140, (3,1), padding='same'))
model.add(Conv2D(84, (3,1), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
# model.save('./model/model_test01.h5')
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])          
# earlystopping = EarlyStopping(monitor='loss', patience=10)
# hist = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2, callbacks=[earlystopping])

# model.save_weights('./model/test_weight1.h5')
# 가중치를 저장하려면 fit 한 다음에
model.load_weights('./model/test_weight1.h5')
# 가중치를 load할 때는 모델이 있어야함
# compile 후에 실행됨
# loss:  0.1263843040773645
# acc:  0.9595000147819519

# 레이어를 하나 더 늘리고 load_weights를 하면 에러가 뜸
# ValueError: You are trying to load a weight file containing 5 layers into a model with 6 layers.

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)
print("loss: ", loss)
print("acc: ", acc)
y_pred = model.predict(x_test[0:10])
y_pred = np.argmax(y_pred, axis=1)
# print(y_test[0:10])
# print(y_test[])
