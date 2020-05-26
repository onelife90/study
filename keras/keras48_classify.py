# classify 분류하다
# 이진분류

#1. 데이터
import numpy as np

x = np.array(range(1,11))
y = np.array([1,0,1,0,1,0,1,0,1,0])
x_predict = np.array([1,2,3])

# 그래프를 그려보면 선을 그을 수 없다. 결과값이 2가지뿐(0 or 1)
# ex) x가 12라면 y의 예측값은? 0 or 1
# 이진분류. binary 분류
# model.add 마지막 레이어에 activation=sigmoid / 0~1 사이의 값으로 출력
# model.compile loss='binary_crossentropy'

# print(x.shape)
# print(y.shape)

# 회귀 모델 생성

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
# activation='relu' 평타 85점
model.add(Dense(9, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(8000))
# activation 디폴트 값 적용
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# activation 전지전능하셔서 모든 레이어에 강림
# 마지막 레이어에서 y=wx+b의 w*sigmoid*마지막 레이어

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])
              # 이진분류에서는 loss 지표가 단 하나뿐! 외우삼 loss='binary_crossentropy'
              # metrics=['acc']를 쓰는 이유? 분류모델의 평가지표
model.fit(x,y, epochs=100, batch_size=1) 

#4. 평가, 예측
loss, acc = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x_predict)

print('loss: ', loss)
print('acc: ', acc)
print('y_predict: \n', y_predict)
#y_predict:
#  [[0.4997568]
#  [0.4997568]
#  [0.4997568]]
# y_predict 값은 sigmoid를 거치지 않았음
