from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])        #(4, )
#from numpy import array를 상단에서 했기 때문에 x = np.array를 안씀
# y2 = array([[4,5,6,7]])             #(1,4)
# y3 = array([[4], [5], [6], [7]])    #(4,1)

print("x.shape : ", x.shape)        #(4,3)
print("y.shape : ", y.shape)        #(4,)

# x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)    #x.shape[0]=4, x.shape[1]=3, 1
# 마지막에 1을 추가==(4,3)에 있던 열작업을 1개씩 하겠다
'''
★ 완전 암기 ★

                행,         열,     몇개씩 자르는지
x의 shape=(batch_size, timesteps, feature)
input_shape = (timesteps, feature)
행무시 input_length = timesteps, input_dim = feature

ex) 14.py 파일
x = np.array([range(1,101), range(311,411), range(100)])
input_dim = 3 ==> batch_size가 행으로 짤림

|___________x___________|
|___1___|__311__|___0___|
|___2___|__312__|___1___|
|___3___|__313__|___2___|
|   .   |   .   |   .   |
|___.___|___.___|___.___|
|__100__|__410__|__90___|

'''
print(x.shape)          #(4,3,1)

#2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3, 1)))      #(4,3,1)에서 행무시 즉,(3,1)
model.add(LSTM(5, input_length=3, input_dim=1))        
# input_shape=(3,1) ==> input_length=3(3개의 컬럼), input_dim=1((3,1)에서 하나씩 잘라서 입력하기 때문에)
# 연산(h)이 다음 연산에 영향을 미친다
# 1. 1*10 = input_dim*1st 노드
# 2. 1*10 = bias*1st 노드
# 3. 10*10 = 역전파 1st 노드를 왔다갔다 하면서 연산
# 4. (1.+2.+3.)*4 = 인풋 게이트, 망각 게이트, 셀 게이트, 아웃풋 게이트 ==> 4배 연산
model.add(Dense(5))
model.add(Dense(1))

model.summary()
# LSTM은 Dense 모델과 비교해도 말도 안되게 많은 연산을 함. 그만큼 자원이 소모됨 --> GPU 연산 준비

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x,y, epochs=500)

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
