# reuters 신문 기사 카테고리

from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)
# 가장 많이 쓰이는 단어 10000개를 load하겠다
print("x_train.shape, x_test.shape: ", x_train.shape, x_test.shape) 
# x_train.shape, x_test.shape:  (8982,) (2246,)
print("y_train.shape, y_test.shape: ", y_train.shape, y_test.shape)
# y_train.shape, y_test.shape:  (8982,) (2246,)
print("첫번째 훈련용 뉴스 기사: \n", x_train[0]) 
# 인덱스 숫자만 리스트 형태로 출력
print(" 첫번째 훈련용 뉴스 기사 레이블: \n", y_train[0])
# 인덱스만 출력

# x_train에 들어있는 숫자들이 각각 어떤 단어들을 나타내는지 확인
word_index = reuters.get_word_index()
print("x데이터의 word_index: \n", word_index)
# 딕셔너리 형태로 각 단어별 인덱스 출력

# 인덱스를 단어로 바꿔주기
from keras.preprocessing.text import Tokenizer
token = Tokenizer()
word_index = token.fit_on_texts(reuters.get_word_index())
word = token.sequences_to_texts(x_train[0:1])
print("x_train의 첫번째 word: \n", word)

# x_train의 shape를 확인하고 싶다?
# 하지만 리스트 형이라 shape가 없다
print(len(x_train[0]))  # 87

# y의 카테고리 개수 출력
category = np.max(y_train) +1
print("y데이터의 레이블 개수: ", category) # 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print("y데이터의 분포: \n", y_bunpo)    # 0~45
# y데이터의 분포:
#  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# y_train의 그룹별 개수 확인
y_train_pd = pd.DataFrame(y_train)
print(y_train_pd)
'''
y_group = y_train_pd.groupby(0).count()
#  Empty DataFrame

y_group = y_train_pd.groupby()[0].count()
# TypeError: You have to supply one of 'by' and 'level'
'''
y_group = y_train_pd.groupby(0)[0].count()
# groupby(기준열)[컬럼명]
print("y의 그룹화: \n", y_group)
print(y_group.shape)    # (46,)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
# print(len(x_train[0]))  # 100
# print(len(x_train[-1])) # 100
print("x_train.shape, x_test.shape: ", x_train.shape, x_test.shape) 
# x_train.shape:  (8982, 100),  x_test.shape:  (2246, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM

model = Sequential()
# model.add(Embedding(1000, 128, input_length=100))
model.add(Embedding(10000, 128))
model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)
# acc:  0.6424754858016968

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
