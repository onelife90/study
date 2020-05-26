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

# one-hot 인코딩
# 0 1 2 3 4 5
#[0 1 0 0 0 0] == 1
#[0 0 1 0 0 0] == 2
#[0 0 0 1 0 0] == 3
#[0 0 0 0 1 0] == 4
#[0 0 0 0 0 1] == 5
# 각 레이블의 인덱스 자리에 맞게 1 표시. 나머지 0인 벡터
# 배열로 만들기 때문에 2차원으로 변환
# 두 가지 이상의 분류면 반드시 one-hot 인코딩!!!

from keras.utils import np_utils
y = np_utils.to_categorical(y)
# category 분류해준다

# print(y)
# print(y.shape)          # (10,6)
# 우리가 의도하는 건 (10,5) 왜? print(y)의 0번째 열이 필요없다.
y = y[:, 1:6]
# print(y)
# print(y.shape)          # (10,6)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='softmax'))
# 다중분류의 활성화함수는 softmax뿐

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
