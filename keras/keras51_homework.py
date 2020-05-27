#  to_categorical은 무조건 [0]번째 인덱스로 시작

# 2번의 첫번째 답

# x = [1,2,3]
# x = x - 1
# print(x)        # type error

import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1
print(y)          # [0,1,2,3,4,0,1,2,3,4]
# 각각의 y값이 

# from keras.utils import np_utils
# y = np_utils.to_categorical(y)
# print(y)
# print(y.shape)

# 2번의 두번째 답

from sklearn.preprocessing import OneHotEncoder
# 사이킷런에서는 차원을 맞춰줘야함
import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y.reshape(-1,1)
print(y.shape)
y = y.reshape(10,1)
print(y.shape)

aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()

print(y)
print(y.shape)

# one-hot 인코딩의 두가지 방법
# 1. from keras.utils import np_utils
# aa = np_utils.to_categorical(a)
# 2. from sklearn.preprocessing import OneHotEncoder
# 차원을 맞춰줘야해서 reshape
