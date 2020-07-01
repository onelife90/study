import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

'''
#2. 모델구성
model = Sequential()
model.add(Conv2D(100, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(80, (3,3), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(300, (3,3), padding='same'))
model.add(Conv2D(2, (3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# model.save('./model/model_test01.h5')
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])          
earlystopping = EarlyStopping(monitor='loss', patience=10)
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2, callbacks=[earlystopping, checkpoint])

# checkpoint 한 결과
# loss:  0.10915890090633183
# acc:  0.9670000076293945

model.save('./model/model_test01.h5')
'''

from keras.models import load_model
model = load_model('./model/10-0.1270.hdf5')
# checkpoint 가설1. 모델과 가중치가 저장되어있다면? 바로 evaluate가 실행됨.
# loss:  0.10915890090633183
# acc:  0.9670000076293945

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)
print("loss: ", loss)
print("acc: ", acc)
y_pred = model.predict(x_test[0:10])
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_test[0:10])
# print(y_test[])
