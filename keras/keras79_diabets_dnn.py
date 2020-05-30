import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# diabets 당뇨병. 이진분류

#1. 데이터
diabets = load_diabetes()
x = diabets.data
y = diabets.target
# print(x[0])         # 10개 컬럼 
# print(y[9])         # 한가지 값. 이진분류기 때문에
# print(x.shape)      # (442,10)
# print(y.shape)      # (442, )

#1-1. 데이터 전처리
y = y.reshape(-1, 1)
# print(y.shape)          # (442,1)

scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x[0])      

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, train_size=0.6)

#2. 모델
input1 = Input(shape=(10, ))
dense1 = Dense(1000)(input1)
dense1 = Dense(700)(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(300)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.1)(dense1)
output1 = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, save_best_only=True, mode='auto')
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stop, checkpoint, tb_hist])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

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

# 튜닝
# epochs=100,batch=1,노드=1000,700,drop0.1,300,100,drop0.1
# 쓰렉..
