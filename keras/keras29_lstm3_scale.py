from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = array([50,60,70])            

print("x.shape : ", x.shape)        # (13,3)
print("y.shape : ", y.shape)        # (13,)
print("x_predict.shape : ", x_predict.shape)       # (3,)

x = x.reshape(x.shape[0], x.shape[1], 1)   
x_predict = x_predict.reshape(1,3,1)

print(x.shape)          #(13,3,1)
print(x_predict.shape)  #(1,3,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, input_length=3, input_dim=1))        
model.add(Dense(5000))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x,y, epochs=500)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

# 하이퍼파라미터튜닝
# epochs=500, LSTM노드=10,5000,1
# [[74.82469]]

# epochs=500, LSTM노드=10,5000,1000,1
# [[74.86908]]

# epochs=500, LSTM노드=10,5000,30,1
# [[75.56215]]

# epochs=1000, LSTM노드=10,5000,30,100,80,1
# [[77.03949]]

# epochs=1000, LSTM노드=10,5000,300,100,3000,80,1
# [[73.995544]]

# epochs=1000, LSTM노드=10,65,30,10,85,5,1
# [[77.456024]]

# epochs=1000, LSTM노드=10,650,30,100,850,5,1
# [[76.226494]]

# epochs=800, LSTM노드=10,650,30,100,850,5,1
# [[78.80027]]

# epochs=500, LSTM노드=10,65,30,100,85,5,1
# [[77.50583]]

# epochs=500, LSTM노드=10,65,30,100,85,5,1
# [[77.50583]]
