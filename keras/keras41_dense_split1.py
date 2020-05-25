import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
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
# print(dataset.shape)
# print(type(dataset))        # numpy.ndarray
# 왜? split_x 함수에서 리턴을 np.array로 했기 때문에

x = dataset[:, 0:4]     # [행, 열] = [: all 모든 값, 0:4] 
y = dataset[:, 4]

print(x)
print(y)

# x = np.reshape(x, (6,4,1))
# x = x.reshape(6,4,1)과 같은 표현

#2. 모델구성

model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(1))

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=30, batch_size=1, verbose=1,
        callbacks=[early_stopping])
# model.fit의 batch_size와 x데이터의 batch_size는 다르다
# x데이터의 batch_size는 총 6행으로 자르겠다는 의미이고
# model.fit의 batch_size는 그 6행을 하나씩 자르겠다는 의미

#4. 평가, 예측

loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: ', y_predict)

# 하이퍼파라미터튜닝
# epochs=30 노드=10,5,1
# loss:  1.547431772109121
# mse:  1.5474318265914917
# y_predict:
#  [[ 2.9187512]
#  [ 4.6222243]
#  [ 6.3256984]
#  [ 8.02917  ]
#  [ 9.732643 ]
#  [11.436116 ]]

# epochs=10 노드=10,5,5,5,5000,5,5,5,1
# loss:  0.12711681957201412
# mse:  0.12711681425571442
# y_predict:
#  [[5.5792103]
#  [6.3741026]
#  [7.168994 ]
#  [7.9638844]
#  [8.758775 ]
#  [9.553666 ]]

# epochs=192 노드=10,5,900,5,1
# loss:  7.958078640513122e-13
# mse:  7.958078640513122e-13
# y_predict:
#  [[ 4.999999]
#  [ 5.999999]
#  [ 7.      ]
#  [ 7.999999]
#  [ 9.      ]
#  [10.000002]]
