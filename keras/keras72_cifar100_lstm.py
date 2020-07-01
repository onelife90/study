import numpy as np
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape)        # (50000, 32, 32, 3)
# print(x_test.shape)         # (10000, 32, 32, 3)
# print(y_train.shape)        # (50000, 1)
# print(y_test.shape)         # (10000, 1)

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)        # (50000, 100)
# print(y_test.shape)         # (10000, 100)

x_train = x_train.reshape(-1,32*32,3).astype('float32')/255
x_test = x_test.reshape(-1,32*32,3).astype('float32')/255

# print(x_train.shape)            # (50000, 1024, 3)
# print(x_test.shape)             # (10000, 1024, 3)

#2. 모델구성
input1 = Input(shape=(1024,3))
dense1 = LSTM(32)(input1)
dense1 = Conv2D(64, (2,2), padding='same')(dense1)
dense1 = Conv2D(128, (2,2), padding='same')(dense1)
dense1 = Conv2D(256, (2,2), padding='same')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = MaxPooling2D(pool_size=3)(dense1)
dense1 = Conv2D(224, (2,1), padding='same')(dense1)
dense1 = Conv2D(175, (2,1), padding='same')(dense1)
dense1 = Conv2D(105, (2,1), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(100, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
modelpath = './model/cifar100/{epoch:02d}-{val_loss:.4f}.hdf5'       
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlystopping, checkpoint])

#4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)

loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

#5. 시각화
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
