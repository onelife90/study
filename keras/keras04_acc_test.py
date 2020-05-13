#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) # 라인 복사 : ctrl+c / 라인 삭제 : shift+del
x_pred = np.array([11,12,13]) # pred = predict

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
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #metrics에는 대괄호 필수 문법임. 훈련 돌릴 때 보여지는 부분
model.fit(x, y, epochs=30, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x, y) #평가 반환 값을 loss, acc(변수)에 넣겠다 #metrics<acc<evaluate
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_pred) #y_pred로 반환한다
print("y_predict : \n", y_pred)

#y_predict : 완벽한 11,12,13 값이 아니다
#  [[11.001056]
#  [12.001145]
#  [13.001239]]

# #why? 회귀와 분류방식
# 1. 회귀 : 수치를 넣었을 때 수치로 답을 한다
# 2. 분류 : 강아지/고양이 _0과 1_ y값이 정해져있어야 함 ex) 동전을 던지고 난 결과
#loss=mse, metrics=acc 이기 때문에 잘못된 모델을 쓴것임
