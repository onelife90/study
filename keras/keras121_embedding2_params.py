from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화에요", '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', 
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])
# print(labels.shape)     # (12,)

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print("token.word_index: \n", token.word_index)
# token.word_index: 
#  {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추
# 천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요
# ': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}
# 중복 제외된 인덱싱. 

# '참'이라는 단어를 3번 주면?
# 많이 사용하는 단어가 인덱싱 우선순위가 됨
# token.word_index:
#  {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에
# 요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}

x = token.texts_to_sequences(docs)
print("token.texts_to_sequences: \n", x)
# 문자를 수치화
# token.texts_to_sequences:
#  [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
# 문제점? shape가 동일 하지 않는 점
# shape를 맞춰줘야함. 하나하나를 reshape 할 수 없음
# pad_sequences를 써준다면! padding을 쓰면 빈자리에 0을 넣어서 진행
# 제일 큰 shape의 숫자를 맞춰서 나머지는 0으로 채우면 동일한 shape로 됨
# LSTM의 경우 : 의미 있는 인덱싱이 뒤로 가는 것이 좋을 수 있음

from keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre')
print("pad_sequences_pre: \n", pad_x)
# pad_sequences_pre:
#  [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]]
# padding='pre' 앞에서부터 0

# pad_x = pad_sequences(x, padding='post')
# print("pad_sequences_post: \n", pad_x)
# pad_sequences_post:
#  [[ 2  3  0  0  0]
#  [ 1  4  0  0  0]
#  [ 1  5  6  7  0]
#  [ 8  9 10  0  0]
#  [11 12 13 14 15]
#  [16  0  0  0  0]
#  [17  0  0  0  0]
#  [18 19  0  0  0]
#  [20 21  0  0  0]
#  [22  0  0  0  0]
#  [ 2 23  0  0  0]
#  [ 1 24  0  0  0]]
# padding='post' 뒤에서부터 0

# pad_x = pad_sequences(x, value=1.0)
# print("pad_sequences_value: \n", pad_x)
# pad_sequences_value:
#  [[ 1  1  1  2  3]
#  [ 1  1  1  1  4]
#  [ 1  1  5  6  7]
#  [ 1  1  8  9 10]
#  [11 12 13 14 15]
#  [ 1  1  1  1 16]
#  [ 1  1  1  1 17]
#  [ 1  1  1 18 19]
#  [ 1  1  1 20 21]
#  [ 1  1  1  1 22]
#  [ 1  1  1  2 23]
#  [ 1  1  1  1 24]]
# value=1.0는 0이 아닌 value 값으로 채워짐

# embedding : 원핫인코딩의 압축형이라 생각하면 됨
# 좌표값만 찍어주면 방대한 데이터를 원핫인코딩을 안해줘도 됨
# 모델 구성시 머신이 수치를 벡터화 해서 임베딩 진행

word_size = len(token.word_index) +1
print("전체 토큰 사이즈 : ", word_size)
# 전체 토큰 사이즈 :  25

#2. 모델 구성을 해보자
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
# model.add(Embedding(word_size, 10, input_length=5))
# 전체 단어의 수, 아웃풋(임베딩 벡터 크기), 인풋
model.add(Embedding(25, 10, input_length=5))   # (None, 5, 10)  3차원으로 변환
# Embedding의 파라미터 계산 = 전체단어의수(조정가능)*아웃풋
# model.add(Embedding(25, 10))     
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
# word_size를 250으로 바꿔준다면?
# 벡터화 주는 크기를 우리가 조정 가능
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 5, 10)             250
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 50)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 51
# =================================================================
# Total params: 301
# Trainable params: 301
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
# model.evaluate에서 반환되는 것은 [0]번째 loss와 [1] metrics
print("acc: ", acc)
# acc:  1.0
