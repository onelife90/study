#1. imdb 검색해서 데이터 내용 확인
# imdb : 긍정(0)과 부정(1)로 라벨링된 영화 데이터셋
#2. word_size 전체데이터 부분에서 최상값 확인
#3. 주간과제 : groupby()의 사용법 숙지할 것
#4. 인덱스를 단어로 바꿔주는 함수 찾을 것

from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000)#, test_split=0.2)
# 가장 많이 쓰이는 단어 2000개를 load하겠다
print("x_train.shape, x_test.shape: ", x_train.shape, x_test.shape) 
# x_train.shape, x_test.shape:  (25000,) (25000,)
print("y_train.shape, y_test.shape: ", y_train.shape, y_test.shape)
# y_train.shape, y_test.shape:  (25000,) (25000,)

print("첫번째 훈련용 영화리뷰: \n", x_train[0]) 
# 인덱스 숫자만 리스트 형태로 출력
print(" 첫번째 훈련용 영화리뷰 레이블: \n", y_train[0])
# 인덱스만 출력

# x_train의 shape를 확인하고 싶다?
# 하지만 리스트 형이라 shape가 없다
print(len(x_train[0]))  # 218

# y의 카테고리 개수 출력
category = np.max(y_train) +1
print("y데이터의 레이블 개수: ", category) # 2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print("y데이터의 분포: \n", y_bunpo)    # 0~1 

# y_train의 그룹별 개수 확인
y_train_pd = pd.DataFrame(y_train)
print(y_train_pd)
y_group = y_train_pd.groupby(0)[0].count()
# groupby(기준열)[컬럼명]

# groupby()의 사용법
#1) 전체 데이터를 그룹 별로 나누고 (split)
#2) 각 그룹별로 집계함수를 적용(apply) 한후
#3) 그룹별 집계 결과를 하나로 합치는(combine) 단계

print("y의 그룹화: \n", y_group)
# y의 그룹화: # 0과 1로 정확히 12500개씩 나뉘어져있음
# 0    12500
# 1    12500
print(y_group.shape)    # (2,)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=1000, padding='pre')
x_test = pad_sequences(x_test, maxlen=1000, padding='pre')
# print(len(x_train[0]))  # 1000
# print(len(x_train[-1])) # 1000
print("x_train.shape, x_test.shape: ", x_train.shape, x_test.shape) 
# x_train.shape, x_test.shape:  (25000, 1000) (25000, 1000)     

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, Conv1D, MaxPooling1D

model = Sequential()
model.add(Embedding(2000, 128))
model.add(Conv1D(64, 5, activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 128)         256000
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, None, 64)          41024
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, None, 64)          0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 64)                33024
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 330,113
# Trainable params: 330,113
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)

#5. 시각화
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='r', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='b', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
