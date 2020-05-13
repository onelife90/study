#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim = 1))
model.add(Dense(30))
model.add(Dense(28))
model.add(Dense(26))
model.add(Dense(24))
model.add(Dense(22))
model.add(Dense(20))
model.add(Dense(18))
model.add(Dense(16))
model.add(Dense(14))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x2)
print(y_predict)


# [[3.9998376]
#  [4.99978  ]
#  [5.999724 ]]
# 레이어 12개
# 1st 노드 = 100
# 노드값 -10


# [[4.000032 ]
#  [5.0000534]
#  [6.000077 ]]
# 레이어 12개
# 1st 노드 = 100
# 2nd 노드 = 2
# 노드값 +2


# [[3.9999986]
#  [5.       ]
#  [5.999998 ]]

# 레이어 14개
# 1st 노드 = 100
# 2nd 노드 = 2
# 노드값 +2


# [[3.9999998]
#  [4.999999 ]
#  [6.000002 ]]
# 레이어 19개
# 1st 노드 = 100
# 2nd 노드 = 30
# 노드값 +2


# [[4.000002]
#  [5.000005]
#  [6.000005]]
# 레이어 19개
# 1st 노드 = 100
# 2nd 노드 = 30
# 노드값 -2
