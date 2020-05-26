# 다중분류

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array(range(1,11))
y = np.array([1,2,3,4,5,1,2,3,4,5])
# 데이터전처리가 필요. 벡터인 2차원으로 변환
x_predict = np.array([1,2,3])

# print(x.shape)
# print(y.shape)

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

print(y)
print(y.shape)      # (10,6)
# 우리가 의도하는 건 (10,5) 왜? print(y)의 0번째 열이 필요없다.
print(type(y))      # <class 'numpy.ndarray'>

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(8000))
model.add(Dense(5, activation='relu'))
model.add(Dense(6, activation='softmax'))
# 다중분류의 활성화함수는 softmax뿐

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
y_predict = model.predict(x_predict)

print('loss: ', loss)
print('acc: ', acc)
print('y_predict: \n', y_predict)
# y_predict:
#  [[0.06832937 0.18680668 0.18600237 0.18575802 0.18661039 0.18649308]
#  [0.06832937 0.18680668 0.18600237 0.18575802 0.18661039 0.18649308]
#  [0.06832937 0.18680668 0.18600237 0.18575802 0.18661039 0.18649308]]
# (10,6)으로 6개가 지정이 되어있기 때문에 이렇게 출력
# 가장 높은 값을 제외한 두번째 큰 숫자가 값을 표현

# 과제 dim을 6->5 변경
# y_predict를 숫자로 바꿔라
