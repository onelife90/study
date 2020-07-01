import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
# print(x[0])         # [5.1 3.5 1.4 0.2] == 4컬럼
# print(x.shape)      # (150, 4)
# print(y.shape)      # (150, )

#1-1. 데이터 전처리
y = np_utils.to_categorical(y)
# print(y.shape)      # (150, 3)

x = x.reshape(-1, 2, 2, 1)
# print(x.shape)        # (150,2,2,1)

#1-2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, train_size=0.8)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(4, (2,1), input_shape=(2,2,1)))
model.add(Conv2D(8, (2,1), padding='same'))
model.add(Conv2D(16, (2,1), padding='same'))
model.add(Conv2D(24, (2,1), padding='same'))
model.add(Conv2D(32, (2,1), padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(28, (2,1), padding='same'))
model.add(Conv2D(20, (2,1), padding='same'))
model.add(Conv2D(12, (2,1), padding='same'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/iris/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stop, checkpoint, tb_hist])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss: ", loss)
print("acc: ", acc)

loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

# print('acc: \n', acc)
# print('val_loss: \n', val_loss)
# print('val_acc: \n', val_acc)

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
# epochs=100, batch=1, 5,1000,max2, drop0.2,30,flat
#loss:  0.06780672576078359
#acc:  0.9833333492279053
