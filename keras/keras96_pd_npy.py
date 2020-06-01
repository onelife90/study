import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

datasets = pd.read_csv("./data/csv/iris.csv", index_col=None, header=0, sep=',')
# index_col=None 읽기 전 파일에 index_col에 데이터가 껴있었다. 그래서 None
# header=0을 하면 첫 헤더(행)는 실 데이터로 인식 X
# print(datasets)

# print(datasets.head())      # 위에서부터 5개
# print(datasets.tail())      # 아래서부터 5개

print("======================")
# print(datasets.values)      # ★항상 쓰임. 머신을 돌리기 위해서 np로 변환

aaa = datasets.values
# print(type(aaa))            # <class 'numpy.ndarray'>

# np로 저장하시오

from sklearn.datasets import load_iris

iris = load_iris()
np.save('./data/iris_all.npy', arr=iris)

# 95번을 불러와서 모델을 완성하시오

#1. 데이터
iris_load=('./data/iris_all.npy')

#1-1. 데이터 전처리
x = aaa[:, 0:4]
y = aaa[:, 4]

# print(x[10])        # [5.4 3.7 1.5 0.2]
# print(y[40])        # 0.0
# print(x.shape)      # (150, 4)
# print(y.shape)      # (150,)

y = np_utils.to_categorical(y)
# print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=99, shuffle=True)

# print(x_train.shape)        # (90, 4)
# print(x_test.shape)         # (60, 4)
# print(y_train.shape)        # (90, 3)
# print(y_test.shape)         # (60, 3)

#2. 모델 구성
input1 = Input(shape=(4, ))
dense1 = Dense(100, activation='softmax')(input1)
dense1 = Dense(80)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(5)(dense1)
output1 = Dense(3, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stop = EarlyStopping(monitor='loss', patience=4, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False)
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[early_stop, checkpoint, tb_hist])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
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
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.title('acc')
plt.legend()
plt.show()

# 튜닝
# epochs=16,batch=1,노드=
#loss:  0.11906453711269478
#acc:  0.9333333373069763

# epochs=22,batch=1,노드=LSTM10,LSTM5,drop0.1,80,60,drop0.1,30,10,5
#loss:  0.18599484871398697
#acc:  0.9666666388511658
