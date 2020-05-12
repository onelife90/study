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

#epochs=500, batch_size=1, Dense=1
#loss : 1.10..(2nd 훈련)
#acc : 1.0

#epochs=1000, batch_size=1, Dense=1
#loss : 5.31..(1st 훈련) / 0.0(3rd 훈련)
#acc : 1.0

#epochs=5000, batch_size=10, Dense=1
#loss : 38.5..(1st 훈련)
#acc : 0.0
#왜 실패지? -> 과적합이라는 부분. 데이터 양이 적은데 훈련을 많이해서 적합치를 넘김!

#epochs=500, batch_size=2, Dense=1
#loss : 1.11..(1st 훈련) / 38.5(2nd 훈련) / 0.057..(3rd)
#acc : 1.0 (1st) / 0.0 (2nd) / 1.0 (3rd)

#epochs=500, batch_size=10, Dense=1
#loss : 38.5 (1st) / 9.010..(2nd 훈련) / 0.155..(3rd) / 38.5(4th)
#acc : 1.0 (1st) / 0.1 (2nd) / 0.7 (3rd) / 0.0 (4th)
