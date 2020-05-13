#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# 라인 복사 : 커서 올려둔 채로 ctrl+c / 라인 삭제 : shift+del

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1)) # activation에도 디폴트가 있다
model.add(Dense(3))
# model.add(Dense(1000000)) #전체 주석 : 블록처리 후 ctrl+/
# model.add(Dense(1000000)) #CPU에서는 백만제곱부터 먹히지 않음
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #metrics에는 대괄호 필수 문법임
model.fit(x, y, epochs=30, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x, y) #평가 반환 값을 loss, acc(변수)에 넣겠다
#metrics<acc<evaluate
print("loss : ", loss)
print("acc : ", acc)

# loss : 0.003(1) / 2.85(2) / 0.009(3) / 2.75(4) / 0.028(5) / 0.033(6) / 3.80(7) / 0.026(8) / 2.35(9) / 2.90 (10)
# acc : 1.0 / 1.0 / 1.0 / 1.0 / 1.0 / 0.5(6) / 1.0 / 1.0 / 1.0 / 1.0
#노드를 급격하게 늘리니 acc가 잘 나온다?

#하이퍼파라미터 튜닝
#2nd 히든레이어 노드 300
#loss : 0.005(1) / 0.001(2) / 1.59(3)
#acc : 1.0 / 1.0 / 0.2(3)
