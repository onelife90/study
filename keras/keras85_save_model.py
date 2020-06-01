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
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2, callbacks=[earlystopping])

model.save('./model/model_test01.h5')
# fit 다음에 model save를 함! fit한 결과값이 저장
# loss:  0.13820593581534923
# acc:  0.9574000239372253

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=100)
print("loss: ", loss)
print("acc: ", acc)
y_pred = model.predict(x_test[0:10])
y_pred = np.argmax(y_pred, axis=1)
print(y_test[0:10])
# print(y_test[])
'''
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

### 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))          # 가로 10인치, 세로 6인치
# subplot(2,1,1)==2행 1열의 첫번째 그림
plt.subplot(2, 1, 1)                
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
plt.grid()                                # 모눈종이처럼 보이게
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val loss'])        # legend==엑셀의 범주(?)와 비슷
plt.legend(loc='upper right')             # loc=location / 명시 안해주면 빈 자리에 자동으로 표시
plt.show()

# subplot(2,1,2)==2행 1열의 2번째 그림
plt.subplot(2, 1, 2)                    
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
plt.grid()                              # 모눈종이처럼 보이게
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['loss', 'val_acc'])        # legend==엑셀의 범주(?)와 비슷
plt.show()
'''
