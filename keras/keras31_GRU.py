from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU
#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])        #(4, )

print("x.shape : ", x.shape)        #(4,3)
print("y.shape : ", y.shape)        #(4,)

x = x.reshape(x.shape[0], x.shape[1], 1)    #x.shape[0]=4, x.shape[1]=3, 1
print(x.shape)          #(4,3,1)

#2. 모델구성
model = Sequential()
model.add(GRU(15, input_length=3, input_dim=1))        
model.add(Dense(9000))
model.add(Dense(300))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x,y, epochs=5000)

#4. 예측
x_predict = array([5,6,7])            
x_predict = x_predict.reshape(1,3,1)
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
