# 모델 저장을 했으니 불러오자

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5                # timesteps = 4

# LSTM 모델을 완성하시오

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

# print(x)
# print(y)

x = np.reshape(x, (6,4,1))
# x = x.reshape(6,4,1)과 같은 표현

#2. 모델구성
from keras.models import load_model
model = load_model('./model/save_keras44.h5')
# keras44 파일의 아웃풋 노드를 10으로 변경
# 당연히 에러가 남. 해결 방법은? 힌트 Sequential
# 아웃풋 노드를 1로 맞춰주고 레이어명을 새로운 걸로 지정해줘야 함

model.add(Dense(1, name='new1'))
model.summary()

#3. 실행
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

# epochs=131 노드=keras44의 노드
# loss:  0.003015353691201502
# mse:  0.003015353577211499
# y_predict:
#  [[5.0346527]
#  [5.9607716]
#  [6.9886374]
#  [8.063926 ]
#  [9.082881 ]
#  [9.93467  ]]

