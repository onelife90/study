import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

#1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=(28*28)))
model.add(Dense(300))
model.add(Dropout(0.3))
model.add(Dense(8000))
model.add(Dropout(0.5))
model.add(Dense(90))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=600)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=600)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)

# 하이퍼파라미터 튜닝
# epochs=100, batch=600, 노드=100,300,Drop(0.3),8000,Drop(0.5),90,Drop(0.2)
#loss:  0.31105909280478955
#acc:  0.9175999760627747

# epochs=100, batch=600, 노드=10,300,Drop(0.3),8000,Drop(0.5),90,Drop(0.2)
#loss: 0.29873051069676876
#acc: 0.9194999933242798

# epochs=100, batch=600, 노드=10,30,Drop(0.3),800,Drop(0.4),90,Drop(0.2)
#loss: 0.2899704186618328
#acc: 0.9204000234603882

# epochs=100, batch=60, 노드=10,30,Drop(0.3),800,Drop(0.4),90,Drop(0.2)
#loss: 
#acc: 
