import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt

#1. 데이터 load
train = pd.read_csv('./data/dacon/emnist/train.csv', header=0, index_col=0, sep=',')
test = pd.read_csv('./data/dacon/emnist/test.csv', header=0, index_col=0, sep=',')
submit = pd.read_csv('./data/dacon/emnist/submission.csv', header=0, index_col=0, sep=',')

#1-1. 픽셀 컬럼 정규화
x_train_pix = train[[str(i) for i in range(784)]]/255
x_pred_pix = test[[str(i) for i in range(784)]]/255

#1-2. 정규화
x_train = pd.concat((pd.get_dummies(train.letter), x_train_pix),axis=1).values.reshape(-1,81,10,1)
x_pred = pd.concat((pd.get_dummies(test.letter), x_pred_pix),axis=1).values.reshape(-1,81,10,1)
y_train = to_categorical(train['digit'].values)

#1-3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=99)
# print(f'x_train,x_test: {x_train.shape} {x_test.shape}')    
# x_train,x_test: (1638, 81, 10, 1) (410, 81, 10, 1)

# print(f'y_train,y_test: {y_train.shape} {y_test.shape}')    
# y_train,y_test: (1638, 10) (410, 10)

#2. 모델 구성
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,2), padding='same', activation='relu', input_shape=(81,10,1)))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(256, kernel_size=(2,2), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16, kernel_size=(2,1), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, batch_size=1, epochs=10, validation_split=0.2)

#4. 평가, 예측
model.evaluate(x_test,y_test,batch_size=1)
y_pred = np.argmax(model.predict(x_pred), axis=1)

print(f'y_pred:\n{y_pred}')
# print(f'y_pred.shape: {y_pred.shape}')  # y_pred.shape: (20480,)

#5. 시각화
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='_', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='_', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

#6. 제출
submit['digit'] = y_pred
submit.to_csv('./data/dacon/emnist/letter_cnn.csv')
