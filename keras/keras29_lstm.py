# 우리가 여태한 DNN(ANN) Deep, Artificial
# CNN Convolutional 복합적인 ex) 엄청나게 많은 이미지를 분석(쪼갤 때)
# RNN Recurrent 순환
# ex) 1 2 3 4 5 6 7 8 9 // ? 
# ex) 1만, 2만, ..., 5만 // ? 
# 시계열 time series

# RNN의 가장 대표적인 LSTM_Long Short(시계열) Term Memory
# 케글, 해커톤 대회에서 유행하는 프로그램이나 코딩법 유행이 달라짐
# 레거시한 방법(엄청 빠른 속도=레이어가 1개인 모델 속도) -> 텐서플로 -> 케라스 // 성능차이=0.99-->0.991
# RNN의 가장 주요한 3가지 타입=Simple RNN, GRU_한국인이 발명, LSTM(가장 좋은 성능)

# RNN, LSTM, GRU의 이미지 검색해보자
# 책 p.158 [그림7-1]
![lstm](https://user-images.githubusercontent.com/64455878/82438545-3455bd80-9ad4-11ea-9207-d68709c6c56e.gif)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 스칼라 벡터 행렬 텐서

# 스칼라=0차원 텐서. 하나의 숫자를 의미, ndim 0차원, rank=축(axis)
# 벡터=1D 텐서. 숫자의 배열, 딱 하나의 축
# 행렬=2D 텐서. 벡터의 배열, 2개의 축(행과 열). 숫자가 채워진 사각 격자
# 텐서=데이터를 위한 컨테이너. 거의 항상 수치형 데이터를 다루므로 숫자를 위한 컨테이너
# ex)행렬 / 다차원 numpy배열, 텐서에서는 차원=축(axis)

## 연습문제
# [[1,2,3], [1,2,3]]                #(2,3)
# [[[1,2], [4,3]], [[4,5], [5,6]]]  #(2,2,2)
# [[[1],[2],[3]], [[4],[5],[6]]]    #(2,3,1)
# [[[1,2,3,4]]]                     #(1,1,4)
# [[[[1], [2]]], [[[3], [4]]]]      #(1,2,1)

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])                #(4, )  # y_스칼라 = 4,5,6,7   # y_벡터 = 1
# 상단에서 from numpy import array를 명시해서 x = np.array를 안씀
y1 = array([[4,5,6,7]])             #(1,4)
y2 = array([[4,5,6,7]])             #(1,4)
y3 = array([[4], [5], [6], [7]])    #(4,1)

print("x.shape : ", x.shape)        #(4,3)
print("y.shape : ", y.shape)        #(4,)

# x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)    #x.shape[0]=4, x.shape[1]=3, 1
# 마지막에 1을 추가==(4,3)에 있던 열작업을 1개씩 하겠다
print(x.shape)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1)))      #(4,3,1)에서 행무시 즉,(3,1)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x,y, epochs=100)

x_input = array([5,6,7])            
# (3,) 스칼라 3개짜리 벡터1개로 x와 모양이 맞지 않음-->(1,3,1)로 reshape
# ( ,3,1) = 3개짜리 1개씩 작업하겠다. 그럼 행은 어떻게 정할까?
# x_input 3차원. 즉, 다 곱해보면 개수가 나옴. reshape 하기 전과 갯수가 같아야 함. 그래서 행은 1
x_input = x_input.reshape(1,3,1)

print(x_input)

yhat = model.predict(x_input)
print(yhat)
# yhat의 출력값이 왜 하나죠?
# |---x---|--y--|
# |1  2  3|  4  |
# |2  3  4|  5  |
# |3  4  5|  6  |
# |4  5  6|  7  |
# |5  6  7|  ?? |   # model.predict 구간에서 예측되는 y값은 1개

# 하이퍼파라미터튜닝
# epochs=100, 노드=10,5,1
#[[7.237007]]

# epochs=100, 노드=10,15,25,35,45,1
#[[7.876493]]

# epochs=100, 노드=10,15,25,3500,45,1
#[[7.8890524]]

# epochs=1000, 노드=10,9000,50,1
#[[7.987792]]

# epochs=1000, 노드=10,9000,500,1
#[[8.048971]]
