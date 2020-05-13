import numpy as np

# 데이터생성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() #순차적인 모델
model.add(Dense(1, input_dim=1, activation='relu')) #히든 레이어가 없다

# 훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#컴파일은 머신에게 어떤 방식으로 모델을 돌릴 것인지 지정해주는 것
model.fit(x, y, epochs=500, batch_size=1) #모델 실행

loss, acc = model.evaluate(x, y, batch_size=1) #모델 평가

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

#epochs=500, batch_size=100, Dense=1
#loss : 38.5 (1) / 0.01(2) / 0.006(3) / 38.5(4)
#acc : 0.0 (1) / 1.0 (2) / 1.0 (3) / 0.0 (4)
#배치사이즈가 100, 데이터가 10개지만 훈련가능.
#왜? 일단 배치사이즈 디폴트 값은 32.\
#가중치는 mini-batch gradient라는 값의 순서대로 업데이트됨
#32는 LSTM 모델의 기본 구성
