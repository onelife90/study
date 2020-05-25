import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5                # timesteps = 4

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)      
# print(dataset)
# print(dataset.shape)        # (96,5)
# print(type(dataset))        # numpy.ndarray
# 왜? split_x 함수에서 리턴을 np.array로 했기 때문에

x = dataset[:, 0:4]     # [행, 열] = [: all 모든 값, 0:4] 
y = dataset[:, 4]
x_predict = dataset[90:, 0:4]

print(x)
print(y)
print(x_predict)

#실습 1. train, test 분리할 것 (8:2)
#실습 2. 마지막 6개의 행을 predict로 만들고 싶다
#실습 3. validation을 넣을 것 (train의 20%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8
    )  

#2. 모델구성

model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(900))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1,
        validation_split=0.2, shuffle=True, callbacks=[early_stopping])
# model.fit의 batch_size와 x데이터의 batch_size는 다르다
# x데이터의 batch_size는 총 6행으로 자르겠다는 의미이고
# model.fit의 batch_size는 그 6행을 하나씩 자르겠다는 의미

#4. 평가, 예측
loss, mse = model.evaluate(x_test,y_test, batch_size=1)
y_predict = model.predict(x_predict)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: \n', y_predict)

# 하이퍼파라미터튜닝
# epochs=28 노드=10,5,900,5,1
# loss:  0.05398799860849977
# mse:  0.05398799851536751
# y_predict:
#  [[94.67086 ]
#  [95.66801 ]
#  [96.665146]
#  [97.662315]
#  [98.659454]
#  [99.6566  ]]

# epochs=18 노드=10,5,700,5,1
# loss:  0.05056407705124002
# mse:  0.05056407302618027
# y_predict:
#  [[ 95.40063]
#  [ 96.40562]
#  [ 97.41058]
#  [ 98.41557]
#  [ 99.42055]
#  [100.42554]]
