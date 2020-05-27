# 다중분류

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array(range(1,11))
y = np.array([1,2,3,4,5,1,2,3,4,5])
# 데이터전처리가 필요. 벡터인 2차원으로 변환
x_predict = np.array([1,2,3])

# print(x.shape)      # (10,)
# print(y.shape)      # (10,)

y = y-1
print(y)

from keras.utils import np_utils
y = np_utils.to_categorical(y)
# one-hot 인코딩을 하는 이유?
# 현재의 y 값을 예시로 들어 2!=1*2, 3!=1*3,... 2배, 3배가 아니기 때문에 각 인덱스에 해당하는 자리로 표시해줘야 한다
# 그래서 one-hot 인코딩을 하는 것이고 y의 종류에 따라 열이 결정됨

print(y)
print(y.shape)          # (10,6)
# 우리가 의도하는 건 (10,5) 왜? print(y)의 0번째 열이 필요없다.
# y = y[:, 1:6]
# print(y)
# print(y.shape)          # (10,6)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50))
model.add(Dense(5, activation='relu'))
# softmax 거치기 전의 output==0.123 0.234 0.34 .. / 모두 더하면 1
model.add(Dense(5, activation='softmax'))
# 다중분류의 활성화함수는 softmax뿐
# softmax를 거친 후의 output은 [0 0 1 0 0] 이런 식으로 인덱스 자리에 표시
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1) +1

print('loss: ', loss)
print('acc: ', acc)

print(y_predict.shape)              # (3,)
print(type(y_predict))              # <class 'numpy.ndarray'>
print('y_predict: \n', y_predict)
# y_predict:
#  [5 5 5]
