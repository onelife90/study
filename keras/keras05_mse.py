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
loss, mse = model.evaluate(x, y) #평가 반환 값을 loss, mse(변수)에 넣겠다 #metrics<mse<evaluate #loss와 metrics가 동일하기에 반환되는 값이 똑같다
print("loss : ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred) #y_pred로 반환한다
print("y_predict : \n", y_pred)


#y_predict
#  [[11.000024]
#  [12.000018]
#  [13.000012]]

# 이 모델의 잘못된점?
# model.fit에 1~10까지 넣고, model.evaluat에도 훈련된 값이 중복
# 따라서 훈련 데이터와 평가 데이터를 구분해야함!
