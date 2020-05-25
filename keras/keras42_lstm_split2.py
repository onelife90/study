import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5                

# LSTM 모델을 완성하시오

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)      
print(dataset)
print(dataset.shape)        # (96,5)
print(type(dataset))        # numpy.ndarray
# 왜? split_x 함수에서 리턴을 np.array로 했기 때문에

x = dataset[:, 0:4]     # [행, 열] = [: all 모든 값, 0:4] 
y = dataset[:, 4]
x_predict = dataset[len(dataset)-6:, 0:4]
# len(dataset)=96개의 리스트 -6 == 마지막 6행

print(x)
print(y)
print(x_predict)

x = x.reshape(x.shape[0],x.shape[1],1)
# print(x.shape)
x_predict = x_predict.reshape(6,4,1)

#실습 1. train, test 분리할 것 (8:2)
#실습 2. 마지막 6개의 행을 predict로 만들고 싶다
#실습 3. validation을 넣을 것 (train의 20%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8
    )  

#2. 모델구성
model = Sequential()
model.add(LSTM(100, input_shape=(4,1)))
model.add(Dense(65))
model.add(Dense(35))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(1))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1,
        validation_split=0.2, shuffle=True, callbacks=[early_stopping])

#4. 평가, 예측
loss, mse = model.evaluate(x_test,y_test, batch_size=1)
y_predict = model.predict(x_predict)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: \n', y_predict)

# 하이퍼파라미터튜닝
# epochs=38 노드=8,3,1,3,2,4,2,1
# loss:  0.8519117459654808
# mse:  0.8519118428230286
# y_predict:
#  [[93.35192 ]
#  [94.1267  ]
#  [94.88074 ]
#  [95.61367 ]
#  [96.32529 ]
#  [97.015366]]

# epochs=17 노드=100,65,45,25,15,1
# loss:  0.4718885190784931
# mse:  0.47188854217529297
# y_predict:
#  [[94.70844 ]
#  [95.500916]
#  [96.26847 ]
#  [97.010735]
#  [97.727455]
#  [98.41855 ]]

# epochs=30 노드=10,15,115,3,1
# loss:  3.326933650765568
# mse:  3.3269336223602295
# y_predict:
#  [[ 96.276825]
#  [ 97.05501 ]
#  [ 97.81656 ]
#  [ 98.56153 ]
#  [ 99.28992 ]
#  [100.0018  ]]
