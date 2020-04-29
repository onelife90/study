#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(124, input_dim = 1))
model.add(Dense(120))
model.add(Dense(116))
model.add(Dense(112))
model.add(Dense(108))
model.add(Dense(104))
model.add(Dense(100))
model.add(Dense(96))
model.add(Dense(92))
model.add(Dense(88))
model.add(Dense(84))
model.add(Dense(80))
model.add(Dense(76))
model.add(Dense(72))
model.add(Dense(68))
model.add(Dense(64))
model.add(Dense(60))
model.add(Dense(56))
model.add(Dense(52))
model.add(Dense(48))
model.add(Dense(44))
model.add(Dense(40))
model.add(Dense(36))
model.add(Dense(32))
model.add(Dense(28))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100)

#4. 평가 예측
loss, acc = model.evaluate(x, y)
print("acc : ", acc)

y_predict = model.predict(x2)
print(y_predict)


# [[3.999731]
#  [4.993538]
#  [5.987344]]
# 레이어 33개
# 1st 노드 = 60
# 2nd 노드 = 58
# 노드값 -2


# [[4.0188456]
#  [5.013905 ]
#  [6.008966 ]]
# 레이어 47개
# 1st 노드 = 60
# 2nd 노드 = 59
# 노드값 -1
