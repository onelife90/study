#  to_categorical은 무조건 [0]번째 인덱스로 시작

#### 2번의 첫번째 답
# x = [1,2,3]
# x = x - 1
# print(x)        # type error

import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1
print(y)          # [0,1,2,3,4,0,1,2,3,4]
# 각각의 y값이 -1되어 출력

# from keras.utils import np_utils
# y = np_utils.to_categorical(y)
# one-hot 인코딩을 하는 이유?
# 현재의 y 값을 예시로 들어 2!=1*2, 3!=1*3,... 2배, 3배가 아니기 때문에 각 인덱스에 해당하는 자리로 표시해줘야 한다
# 그래서 one-hot 인코딩을 하는 것이고 y의 종류에 따라 열이 결정됨

# print(y)
# print(y.shape)

#### 2번의 두번째 답
from sklearn.preprocessing import OneHotEncoder
# 사이킷런에서는 차원을 맞춰줘야함
import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y.reshape(-1,1)         # reshape(-1,1)의 -1은 자동으로 행을 맞추겠다
print(y.shape)
y = y.reshape(10,1)
print(y.shape)

aaa = OneHotEncoder()       
aaa.fit(y)
y = aaa.transform(y).toarray()
# preprocessing(전처리)로 반드시 fit실행 후 transform변환
# toarray = array 배열 형태로 만들겠다

print(y)
print(y.shape)

# one-hot 인코딩의 두가지 방법
# 1. from keras.utils import np_utils
# aa = np_utils.to_categorical(a)
# 2. from sklearn.preprocessing import OneHotEncoder
# 차원을 맞춰줘야해서 reshape. 그리고 전처리이기 때문에 선실행 후변환 필수!
