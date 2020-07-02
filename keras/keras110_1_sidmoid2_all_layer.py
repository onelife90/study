# sigmoid를 레이어마다 써준다면?
# 활성화 함수는 전지전능하셔서 모든 레이어에 강림하심

#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(100, input_shape=(1,)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# h1(x)=w1*x+w2*x+...+b가 activation에 들어가서 계산

#3. 컴파일, 훈련
model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_train, y_train)
print("loss: ", loss)
# loss:  [-248.4429168701172, 0.10000000149011612]

x1_pred = np.array([11,12,13,14])
y_pred = model.predict(x1_pred)
print("y_pred: \n", y_pred)
# [[1.]
#  [1.]
#  [1.]
#  [1.]]
