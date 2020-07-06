from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화에요", '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', 
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print("token.word_index: \n", token.word_index)
# token.word_index: 
#  {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추
# 천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요
# ': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}

x = token.texts_to_sequences(docs)
print("token.texts_to_sequences: \n", x)
# 문자를 수치화
# token.texts_to_sequences:
#  [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]

from keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre')
print("pad_sequences_pre: \n", pad_x)
# padding='pre' 앞에서부터 0

# embedding : 원핫인코딩의 압축형이라 생각하면 됨
# 좌표값만 찍어주면 방대한 데이터를 원핫인코딩을 안해줘도 됨
# 모델 구성시 머신이 수치를 벡터화 해서 임베딩 진행

word_size = len(token.word_index) +1
print("전체 토큰 사이즈 : ", word_size)
# 전체 토큰 사이즈 :  25

#2. 모델 구성을 해보자
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
# model.add(Embedding(word_size, 10, input_length=5))
# 전체 단어의 수, 아웃풋(임베딩 벡터 크기), 인풋

model.add(Embedding(25, 10))     # (None, None, 10)
# Embedding의 파라미터 = 25*10 
# Embedding과 LSTM을 함께 쓰면 input_length를 명시 안해도 됨
model.add(LSTM(3))
# Embedding의 아웃풋이 LSTM의 인풋이 됨
# 4*(3+1+10)*3 = 168

# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 10)          250
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 3)                 168
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 422
# Trainable params: 422
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
