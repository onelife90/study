# keras56 땡겨서 실습
# input_dim=154로 모델 생성

import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()
# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255
#print(x_train.shape)        # (60000, 784)
#print(x_test.shape)         # (10000, 784)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)
# print(y_test.shape)         # (10000, 10)

#1-2. X데이터 train+test 합치기
X = np.append(x_train, x_test, axis=0)
print(f'X.shape: {X.shape}')    # X.shape: (70000, 784)

#1-3. 차원을 축소하지 않고 PCA 계산 후 최적의 n_components 구하기
#pca = PCA()
#pca.fit(X)
#cumsum = np.cumsum(pca.explained_variance_ratio)

#best_n_components = np.argmax(cumsum>=0.95)+1
#print(f'best_n_components: {best_n_components}')
# best_n_components: 154

#1-4. PCA 실행 후 변환
pca = PCA(n_components=154)
X = pca.fit_transform(X)

#1-5. train, test 다시 분리하기
x_train = X[:60000, :]
x_test = X[60000:, :]

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_shape=(154, )))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x_train, y_train, epochs=50, batch_size=256, validation_split=0.2, callbacks=[earlystopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print(f'loss: {loss:.5f}, acc: {acc:5f}')
# loss: 0.28328, acc: 0.921100        
print(f'y_pred: {y_pred}')
