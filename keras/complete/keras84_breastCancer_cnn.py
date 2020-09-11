import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. 데이터
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# print(x[0])         # 30개 컬럼
# print(y)            # 0,1 여러개 다중분류
# print(x.shape)      # (569,30)
# print(y.shape)      # (569, )

y = np_utils.to_categorical(y)
# print(y.shape)      # (596,2)
x = x.reshape(-1,6,5,1)
# print(x.shape)      # (596,6,5,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, train_size=0.8)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(30, (2,2), padding='same', input_shape=(6,5,1)))
model.add(Conv2D(60, (2,2), padding='same'))
model.add(Conv2D(120, (2,2), padding='same'))
model.add(Conv2D(180, (2,2), padding='same'))
model.add(Conv2D(240, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=3, padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(210, (2,2), padding='same'))
model.add(Conv2D(150, (2,2), padding='same'))
model.add(Conv2D(90, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/cancer/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,  write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stop, checkpoint, tb_hist])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

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
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2,1,2)
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()
plt.show()
