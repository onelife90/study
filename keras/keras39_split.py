import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5

# 코드 분석 시, 처음보는 변수명이나 매개변수가 나와도 당황하지 말자
# 코드 안에 반드시 존재한다. 그렇지 않으면 구글링링링링마벨
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
                    # len=요소개수 / seq=a 왜? 18라인에 dataset 변수에서 split_x가 재사용 되기 때문에 seq=a가 됨
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
# print(len(a))
print("==================")
print(dataset)
