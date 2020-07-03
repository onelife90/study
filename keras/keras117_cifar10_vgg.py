# cifar10에 vgg16을 엮어 보시오

import numpy as np
from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from keras.applications import InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)    # (50000, 32, 32, 3)
# print(x_test.shape)     # (10000, 32, 32, 3)
# print(y_train.shape)    # (50000, 1)
# print(y_test.shape)     # (10000, 1)

#1-1. 데이터 전처리
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# y데이터를 원핫인코딩 처리를 할거라면, loss=categorical_crossentropy로 수정
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (50000, 10)
# print(y_test.shape)         # (10000, 10)

#2. 모델구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) 
# vgg16 = VGG16()   #(None,224,224,3)
# vgg16.summary()

model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
# evaluate하면 loss, acc가 출력
print("loss: ", loss)
print("acc: ", acc)

# y_pred = model.predict(x_test)
# print(y_pred)

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

# loss:  0.8188124282836914
# acc:  0.8248000144958496

# loss:  0.7357044310688973
# acc:  0.8465999960899353

# x,y 전처리
# loss:  0.6669038108143955
# acc:  0.8575000166893005
