import numpy as np
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
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

x_train = x_train.reshape(-1,32*32*3).astype('float32')/255
x_test = x_test.reshape(-1,32*32*3).astype('float32')/255

# print(x_train.shape)            # (50000, 3072)
# print(x_test.shape)             # (10000, 3072)

#2. 모델구성
input1 = Input(shape=(3072, ))
dense1 = Dense(800)(input1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.3)(dense1)
output1 = Dense(100, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'       
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlystopping, checkpoint])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss: ", loss)
print("acc: ", acc)

loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

# print('acc: \n', acc)
# print('val_loss: \n', val_loss)
# print('loss_acc: \n', loss_acc)

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

# 튜닝
# epochs=113,batch=100,노드=5000.drop0.5,300,100,drop0.3,30,20,10
