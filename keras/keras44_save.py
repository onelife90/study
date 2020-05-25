# 모델을 저장해보자

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#2. 모델구성

model = Sequential()
model.add(LSTM(8, input_shape=(4,1)))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# model.save("save_keras44.h5")
print("저장 잘됐다.")
# .h5라는 확장자 사용
# 어디에 저장? 경로를 지정해주지 않아서 study 폴더에 저장
# 경로 지정을 해보자

# model.save(".//model//save_keras44.h5")
# model.save("./model/save_keras44.h5")
model.save(".\model\save_keras44.h5")
# .==현재폴더
# 세가지 방법 모두 저장 됨
