import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5
# 10을 6개씩 자르면 (5x6)
# 1 2 3 4 5 6
# 2 3 4 5 6 7 ...
# 즉, (data set 개수-size+1, size)가 됨

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
