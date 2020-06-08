# SimpleRNN도 3차원 구성!

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])        
# print("x.shape : ", x.shape)        #(4,3)
# print("y.shape : ", y.shape)        #(4,)

x = x.reshape(x.shape[0], x.shape[1], 1)    #x.shape[0]=4, x.shape[1]=3, 1
# print(x.shape)          #(4,3,1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(5, input_length=3, input_dim=1))        
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x,y, epochs=1000)

#4. 예측
x_predict = array([5,6,7])            
x_predict = x_predict.reshape(1,3,1)
# x.shape(1,3,1)이기 때문에 x_predict.shape(1,3,1)이 되어야함!
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

# 하이퍼파라미터 튜닝
# epochs=1000 SimpleRNN노드=1,5,5,1
# [[7.324559]]

# epochs=1000 SimpleRNN노드=15,555,1
# [[7.718759]] / [[7.782217]]

# epochs=2000 SimpleRNN노드=15,9000,1
# [[7.8463387]] / [[7.855679]]

# epochs=5000 SimpleRNN노드=15,9000,1
# [[7.8747296]]

# epochs=5000 SimpleRNN노드=15,9000,300,1
# [[7.9178743]] / [[7.9034195]]
