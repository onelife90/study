# 데이터 구성
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터 생성
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

# 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=1,
            validation_data= (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

# 예측
print("loss : ", loss)
print("acc : ", acc)

#하이퍼파라미터 

#epochs=100, batch_size=1, Dense=5,3,1
#loss : 11138(1) / 1.11(2)
#acc : 0.0(1) / 1.0(2)

#epochs=200, batch_size=1, Dense=5,3,1
#loss : 11138(1) / 2.50(2)
#acc : 0.0(1) / 0.0(2)

#epochs=50, batch_size=1, Dense=5,3,1
#loss : 206(1) / 11138(2)
#acc : 0.0(1) / 0.0(2)
#거의 쓰레기값

#epochs=100, batch_size=2, Dense=5,3,1
#loss : 278(1) / 11138(2)
#acc : 0.0(1) / 0.0(2)

#epochs=100, batch_size=2, Dense=10,7,6,5,3,1
#loss : 9.53(1) / 11138(2)
#acc : 0.0(1) / 0.0(2)

#epochs=100, batch_size=1, Dense=10,7,6,5,3,1
#loss : 9.53(1) / 8.14(2) / 11138(3)
#acc : 0.0(1) / 1.0(2) / 0.0(3)

#epochs=50, batch_size=1, Dense=10,7,6,5,3,1
#loss : 3.44(1) / 27(2)
#acc : 0.0(1) / 0.0(2)

#epochs=50, batch_size=1, Dense=14,12,10,8,6,4,2,1
#loss : 4.98(1) / 11138(2)
#acc : 0.0(1) / 0.0(2)

#epochs=80, batch_size=1, Dense=14,12,10,8,6,4,2,1
#loss : 2.96(1) / 1.43(2) / 11138(3)
#acc : 1.0(1) / 1.0(2) / 0.0(3)
#결론 : epochs를 너무 높여도 안되지만 너무 낮추면 쓰레기값이 된다. 결국 경험치?

#epochs=100, batch_size=1, Dense=30,28,26,...14,12,10,8,6,4,2,1
#loss : 11138(1) / 6.28(2) / 7.16(3) / 0.122(4)
#acc : 0.0(1) / 1.0(2) / 1.0(3) / 1.0(4)
#깊이를 추가하니 정확도 올라감

#epochs=100, batch_size=2, Dense=30,28,26,...14,12,10,8,6,4,2,1
#loss : 1.51(1) / 3.43(2) / 11138(3)
#acc : 1.0(1) / 1.0(2) / 0.0(3)
#데이터가 몇개 없어서 배치사이즈를 높여도 별 효과없음

#epochs=50, batch_size=1, Dense=30,28,26,...14,12,10,8,6,4,2,1
#loss : 4.36(1) / 5.41(2) / 11138(3)
#acc : 1.0(1) / 1.0(2) / 0.0(3)
#깊이가 많아도 에포가 반으로 줄어 결국 쓰레기값

#epochs=100, batch_size=1, Dense=랜덤값
#loss : 2.82(1) / 11138(2) / 0.04(3) / 48(4) / 0.00(5) / 1.76(6)
#acc : 1.0(1) / 0.0(2) / 1.0(3) / 0.0(4) / 1.0(5) / 0.0(6)
#덴스가 랜덤이라 그런지 acc값 왓다갔다함
