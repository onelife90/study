#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 라인 복사 : ctrl+c / 라인 삭제 : shift+del
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) # pred = predict

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
model.add(Dense(400))
model.add(Dense(450))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #metrics에는 대괄호 필수 문법임. 훈련 돌릴 때 보여지는 부분
model.fit(x_train, y_train, epochs=30, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test) #평가 반환 값을 loss, mse(변수)에 넣겠다 
#mse<metrics<evaluate 
#loss와 metrics가 동일하기에 반환되는 값이 똑같다
print("loss : ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred) #y_pred로 반환한다
print("y_predict : \n", y_pred)

#하이퍼파라미터 튜닝
#ephocs=30, batch_size=1,Dense=300,280,260...
# y_predict :
# [[16.009815]
# [17.010408]
# [18.010998]]

# train/test
# x=수능점수, 온도, 날씨, 하이닉스, 유가 환율, 금시세, 금리 등
# y=삼성주가
# 예를들어 x의 자료가 엑셀로 365일치의 데이터로 만들어져있다면, train 7달, test 3달로 나누어서 모델을 돌려야 y값이 정확히 나올 수 있다
# model.fit에 train 값을, model.evalatae에 test 값을 넣으면 된다.
# 왜 나누는가?
# ex.수능시험 답만 외운 애들은 수능가서 망함
# 평가 데이터(검증값)는 모델에 반영이 안됨!
