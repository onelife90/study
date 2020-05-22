from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

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

input1 = Input(shape=(3,1))
dense1 = LSTM(10, return_sequences=True)(input1)
# LSTM은 3차원구조 (batch_size, timesteps, feature)로 바꿔줘야함
dense1 = LSTM(10)(dense1)
# LSTM 레이어가 완성되면 출력되는 값은 2차원으로 진행. return_sequences의 디폴트 값은 False이기에
dense1 = Dense(5)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input1                          (n,3,1)
input_1 (InputLayer)         (None, 3, 1)              0
_________________________________________________________________
output노드가 다음 파라미터 연산의 feature로 간다!
feature=input_dim이 되어 4*(output노드+input_dim+bias)*output노드
bias는 레이어 하나에 항상 하나만 존재. 갯수이기 때문에!
lstm_1 (LSTM)                (None, 3, '10')             480
_____________________________________input_dim=feature___________
lstm_2 (LSTM)                (None, 10)                840
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 1,381
Trainable params: 1,381
Non-trainable params: 0
_________________________________________________________________
'''
'''
#3. 훈련
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x,y, epochs=5000, callbacks=[early_stopping])

print(x_predict)

#4. 평가, 예측
y_predict = model.predict(x_predict)
print(y_predict)

loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
'''
