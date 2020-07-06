# 122를 카피해서 conv1d로 구성
# embedding + Conv1D 완성

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

x = token.texts_to_sequences(docs)
print("token.texts_to_sequences: \n", x)
# 문자를 수치화

# 문제점? shape가 동일 하지 않는 점
# shape를 맞춰줘야함. 하나하나를 reshape 할 수 없음
# pad_sequences를 써준다면! padding을 쓰면 빈자리에 0을 넣어서 진행
# 제일 큰 shape의 숫자를 맞춰서 나머지는 0으로 채우면 동일한 shape로 됨
# LSTM의 경우 : 의미 있는 인덱싱이 뒤로 가는 것이 좋을 수 있음

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
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D

model = Sequential()
model.add(Embedding(word_size, 10, input_length=5))
model.add(Conv1D(3, kernel_size=1))
# Conv1D이기 때문에 kernel_size도 1차원으로 넣어줌
# output*(input+1)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 5, 10)             250
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, 5, 3)              33
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 15)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 16
# =================================================================
# Total params: 299
# Trainable params: 299
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
