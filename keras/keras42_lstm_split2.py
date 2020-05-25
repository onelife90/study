import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5                # timesteps = 4

# LSTM 모델을 완성하시오

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)      # (96,5)
# print(dataset)
# print(dataset.shape)
# print(type(dataset))        # numpy.ndarray
# 왜? split_x 함수에서 리턴을 np.array로 했기 때문에

x = dataset[:, 0:4]     # [행, 열] = [: all 모든 값, 0:4] 
y = dataset[:, 4]

x_predict = dataset[-6,-6]
# print(x)
# print(y)

x = np.reshape(x, (6,4,1))
# x = x.reshape(6,4,1)과 같은 표현

#실습 1. train, test 분리할 것 (8:2)
#실습 2. 마지막 6개의 행을 predict로 만들고 싶다
#실습 3. validation을 넣을 것 (train의 20%)

from sklearn.
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, train_size=0.8
)  

#2. 모델구성

model = Sequential()
model.add(LSTM(8, input_shape=(4,1)))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=1000, batch_size=1, verbose=1,
        callbacks=[early_stopping])
# model.fit의 batch_size와 x데이터의 batch_size는 다르다
# x데이터의 batch_size는 총 6행으로 자르겠다는 의미이고
# model.fit의 batch_size는 그 6행을 하나씩 자르겠다는 의미

#4. 평가, 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: \n', y_predict)

# loss:  0.0026065367796945793
# mse:  0.0026065369602292776
# y_predict:
#  [[5.0208592]
#  [6.0008645]
#  [6.9798226]
#  [8.040006 ]
#  [9.089726 ]
#  [9.928272 ]]
