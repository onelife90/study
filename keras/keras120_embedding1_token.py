# 자연어 처리의 기본
# 토큰의 개념부터 알아보자
from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다"
token = Tokenizer()
token.fit_on_texts([text])

print("token.word_index: \n", token.word_index)
# token.word_index:
#  {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}
# 딕셔너리 형태로 텍스트를 단어 단위로 잘라서 인덱싱(수치화)을 걸어줌

x = token.texts_to_sequences([text])
print("token.texts_to_sequences: \n", x)
# token.texts_to_sequences:
#  [[1, 2, 3, 4]]
# 리스트 형태로 텍스트의 인덱스만 반환
# LSTM 모델에 넣고 다음 단어를 예측할 수 있음
# 하지만, 여기서 문제는 2!=1*2, 3!=1*3.. 아니다. 단지 인덱싱일 뿐

# 그럼 원핫인코딩을 해볼까?
from keras.utils import to_categorical

word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)

print("to_categorical: \n",x)
# to_categorical:
#  [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]
# to_categorical.shape=(4,5)
# 여기서 문제점은? 단어가 늘어날수록 데이터가 방대해지는 문제점

# 그렇다면 해결방법은? 압축해보자. 그래서 나온 개념이 embedding
# embedding 자연어처리, 시계열에서 상당히 많이 사용
