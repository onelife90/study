# 전이학습
# 모델 저장을 했으니 불러오자

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

dataset = split_x(a, size)
# print(dataset)
# print(dataset.shape)
# print(type(dataset))        # numpy.ndarray
# 왜? split_x 함수에서 리턴을 np.array로 했기 때문에

x = dataset[:, 0:4]     # [행, 열] = [: all 모든 값, 0:4] 
y = dataset[:, 4]

# print(x)
# print(y)

x = np.reshape(x, (96,4,1))
# x = x.reshape(6,4,1)과 같은 표현

#2. 모델구성
from keras.models import load_model
model = load_model('./model/save_keras44.h5')
# keras44 파일의 아웃풋 노드를 10으로 변경
# 당연히 에러가 남. 해결 방법은? 힌트 Sequential
# The name "dense_1" is used 2 times in the model. All layer names should be unique.
# 아웃풋 노드를 1로 맞춰주고 레이어명을 새로운 걸로 지정해줘야 함

model.add(Dense(170, name='new1'))
model.add(Dense(100, name='new2'))
model.add(Dense(1, name='new3'))
model.summary()

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x,y, epochs=100, batch_size=1, verbose=1,
                 callbacks=[early_stopping], validation_split=0.2)

print(hist)                     # <keras.callbacks.callbacks.History object at 0x0000013432FAC8C8>
# 자료형만 보여줌
print(hist.history.keys())      # dict_keys(['loss', 'mse'])

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

'''
#4. 평가, 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: \n', y_predict)
'''
