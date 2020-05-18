# 함수형 모델
# 함수는 재사용을 위해 생성
# A_sequential 모델 + B_sequential 모델. 어떻게 한번에 묶을까? A_sequential 모델==A 함수, B_sequential 모델==B함수로 묶어줘서 표현

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array(range(711,811))

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, train_size=0.6
)       

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # 함수형 모델은 input, output을 명시해줘야함
# model = Sequential()
# model.add(Dense(5, input_dim = 3))
# model.add(Dense(4))
# model.add(Dense(1))

input1 = Input(shape=(3, )) #행무시 열우선. 100행 3열이기에 3열 표시
# 변수명은 소문자로 암묵적인 룰
# 함수형 모델에서는 keras.layer라는 계층 친구인 Input을 명시해줘야함
dense1 = Dense(5, activation='relu')(input1)
# activation=활성화 함수 # 앞단의 아웃풋이 뒤 꽁지에 붙음
dense2 = Dense(4, activation='relu')(dense1)
output1 = Dense(1)(dense2) #activation에도 디폴트가 있음

model = Model(inputs=input1, outputs=output1)
# 순차적 모델은 model = Sequential()이라고 명시
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시

model.summary()
