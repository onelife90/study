import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. 데이터
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# print(x[0])         # 30개 컬럼
# print(y)            # 0,1로 여러개
# print(x.shape)      # (569,30)
# print(y.shape)      # (569, )

#1-1. 데이터 전처리
y = np_utils.to_categorical(y)
# print(y.shape)      # (569,2)
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)      # (569,30)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, train_size=0.8)

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_shape=(30, )))
model.add(Dense(60, activation='softmax'))
model.add(Dense(120, activation='softmax'))
model.add(Dense(180, activation='softmax'))
model.add(Dense(240, activation='softmax'))
model.add(Dense(210, activation='softmax'))
model.add(Dense(150, activation='softmax'))
model.add(Dense(90, activation='softmax'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
modelpath = './model/cancer/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,  write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[early_stop, checkpoint, tb_hist])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=5)
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

# 튜닝
# epochs=19,batch=5,노드=100,5000,9000,drop0.1,7000,300,100
#loss:  0.6581114022093907
#acc:  0.6315789222717285
