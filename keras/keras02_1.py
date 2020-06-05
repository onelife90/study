# 데이터 구성
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터 생성
x_train = np.array([1,3,5,7,9,11,13,15,17,19])
y_train = np.arraay([11,33,55,77,99,121,143,165,187,209])
x_test = np.array([2,4,6,8,10,12,14,16,18,20])
y_test = np.array([22,44,66,88,110,132,154,176,198,220])

# 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, ephocs=100, batch_size=1,
            validation_data= (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

# 예측
print("loss : ", loss)
print("acc : ", acc)

                   
#하이퍼파라미터 튜닝
                   
#epochs=100, batch_size=1, 노드=5,3,1
#loss : 14(1) / 105(2) / 18634(3)
#acc : 0.1(1) / 0.0(2) / 0.0(3)
#안깊어서?

#epochs=100, batch_size=1, 노드=30,28,26,...2,1
#loss : 18634(1) / 0.001(2) / 8.04(3)
#acc : 0.0(1) / 1.0(2) / 1.0(3)
#깊이를 줬지만 2번째 훈련때만 acc 잘나옴..
                   
#epochs=50, batch_size=1, 노드=30,28,26,...2,1
#loss : 0.42(1) / 0.14(2) / 0.04(3)
#acc : 0.4(1) / 1.0(2) / 1.0(3)
#에포를 반으로 줄이니 손실도 최소화
                 
#epochs=80, batch_size=1, 노드=30,28,26,...2,1
#loss : 18634(1) / 0.3(2) / 0.00019(3)
#acc : 0.0(1) / 1.0(2) / 1.0(3)
#에포를 80으로 맞추고 3rd 훈련을 하니 loss 최소화
                   
#epochs=90, batch_size=1, 노드=30,28,26,...2,1
#loss : 1.55(1) / 2.49(2) / 18634(3)
#acc : 1.0(1) / 1.0(2) / 0.0(3)
#데이터가 10개지만 에포 90은 무리? 80이 최적

#epochs=80, batch_size=1, 노드=랜덤
#loss : 18634(1) / 3.30(2) / 0.1(3) / 18634(4)
#acc : 0.0(1) / 0.2(2) / 0.8(3) / 0.0(4)
#노드 랜덤값은..

#epochs=80, batch_size=1, 노드=15,14,13,...,1
#loss : 18634(1) / 0.016(2) / 0.066(3) / 18634(4)
#acc : 0.0(1) / 1.0(2) / 1.0(3) / 0.0(4)
#레이어의 수를 줄이는 게 더 도움?
