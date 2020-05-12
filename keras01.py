import numpy as np

# 데이터생성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu')) #히든 레이어가 없다

# 훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=1)

loss, acc = model.evaluate(x, y, batch_size=1)

# 예측
print("loss : ", loss)
print("acc : ", acc)

#loss : 1.10..(2nd 훈련)
#acc : 1.0
