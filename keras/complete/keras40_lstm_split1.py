import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5               

# LSTM 모델을 완성하시오

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        # seq[0:5] == [1,2,3,4,5]
        # seq[1:6] == [2,3,4,5,6]
        # seq[2:7] == [3,4,5,6,7]
        # seq[3:8] == [4,5,6,7,8]
        # seq[4:9] == [5,6,7,8,9]
        # seq[5:10] == [6,7,8,9,10]
        aaa.append([item for item in subset])
        # aaa = [[1,2,3,4,5]
        #        [2,3,4,5,6]
        #        [3,4,5,6,7]
        #        [4,5,6,7,8]
        #        [5,6,7,8,9]
        #        [6,7,8,9,10]]
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

# print(dataset)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
# print(dataset.shape)        # (6, 5)
# print(type(dataset))        # numpy.ndarray
# 왜? split_x 함수에서 리턴을 np.array로 했기 때문에

x = dataset[:, 0:4]     # [행, 열] = [: all 모든 값, 0:4] 
y = dataset[:, 4]
#  [0][1][2][3][4]
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

# print(x)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
# print(y)
# [ 5  6  7  8  9 10]

x = np.reshape(x, (6,4,1))
# x = x.reshape(6,4,1)과 같은 표현

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

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=1000, batch_size=1, verbose=1,
         callbacks=[early_stopping])
# model.fit의 batch_size와 x데이터의 batch_size는 다르다
# x데이터의 batch_size는 총 6행으로 자르겠다는 의미이고
# model.fit의 batch_size는 그 6행을 하나씩 자르겠다는 의미
# ex)  1/2/3/4/5
#      2/3/4/5/6
#      3/4/5/6/7
#      4/5/6/7/8
#      5/6/7/8/9
#      6/7/8/9/10

#4. 평가, 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: \n', y_predict)

# 하이퍼파라미터튜닝
# epochs=10 LSTM노드=10 Dense노드=5,5,5000,5,5,1
# loss:  0.024961726701197524
# mse:  0.024961726740002632
# y_predict:
#  [[4.9487615]
#  [6.096883 ]
#  [7.1617827]
#  [8.112181 ]
#  [8.950272 ]
#  [9.689309 ]]

# epochs=10 LSTM노드=10 Dense노드=5,5,5000,5,5,1
# loss:  0.049486996761212744
# mse:  0.04948699474334717
# y_predict:
#  [[4.8464336]
#  [6.1403894]
#  [7.302879 ]
#  [8.296058 ]
#  [9.109151 ]
#  [9.750338 ]]

# epochs=13 LSTM노드=10 Dense노드=35,15,5000,45,25,1
# loss:  0.07884421789397796
# mse:  0.07884421944618225
# y_predict:
#  [[ 4.9015903]
#  [ 6.1818004]
#  [ 7.3602266]
#  [ 8.413879 ]
#  [ 9.335708 ]
#  [10.12873  ]]

# epochs=44 LSTM노드=9 Dense노드=35,15,50,45,25,1
# loss:  0.04030502277358513
# mse:  0.04030502215027809
# y_predict:
#  [[4.7061343]
#  [5.9918447]
#  [7.1480956]
#  [8.160976 ]
#  [9.004843 ]
#  [9.672071 ]]

# epochs=53 LSTM노드=8 Dense노드=35,15,50,45,25,1
# loss:  0.05724603557609953
# mse:  0.05724603310227394
# y_predict:
#  [[4.645985 ]
#  [5.9800434]
#  [7.14594  ]
#  [8.130372 ]
#  [8.936669 ]
#  [9.581139 ]]

# epochs=85 LSTM노드=9 Dense노드=35,15,50,45,25,1
# loss:  0.01076701801988141
# mse:  0.010767017491161823
# y_predict:
#  [[4.8866086]
#  [5.93494  ]
#  [7.0221047]
#  [8.074183 ]
#  [9.018183 ]
#  [9.797049 ]]

# epochs=125 LSTM노드=9 Dense노드=3,1,5,4,2,1
# loss:  0.006131935690063983
# mse:  0.006131935864686966
# y_predict:
#  [[4.976225 ]
#  [5.970655 ]
#  [7.0429835]
#  [8.114174 ]
#  [9.087254 ]
#  [9.886558 ]]

# epochs=132 LSTM노드=9 Dense노드=3,1,1,4,2,1
# loss:  0.03282601929580172
# mse:  0.0328260213136673
# y_predict:
#  [[4.7923183]
#  [6.04455  ]
#  [7.18989  ]
#  [8.210459 ]
#  [9.072996 ]
#  [9.742783 ]]

# epochs=149 LSTM노드=9 Dense노드=3,1,3,4,2,1
# loss:  0.010562797503856322
# mse:  0.010562797077000141
# y_predict:
#  [[4.8913946]
#  [5.9848   ]
#  [7.078879 ]
#  [8.132427 ]
#  [9.081482 ]
#  [9.855249 ]]

# epochs=135 LSTM노드=8 Dense노드=3,1,3,4,2,1
# loss:  0.0026140977758283648
# mse:  0.0026140976697206497
# y_predict:
#  [[4.993496 ]
#  [5.994078 ]
#  [7.0105786]
#  [8.0609   ]
#  [9.074264 ]
#  [9.920806 ]]

# epochs=131 LSTM노드=8 Dense노드=3,1,3,2,4,2,1
# loss:  0.0026065367796945793
# mse:  0.0026065369602292776
# y_predict:
#  [[5.0208592]
#  [6.0008645]
#  [6.9798226]
#  [8.040006 ]
#  [9.089726 ]
#  [9.928272 ]]
