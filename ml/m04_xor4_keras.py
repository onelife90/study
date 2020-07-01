# xor 모델을 keras로 완성하시오.
# 조건1. 딥러닝에서 딥을 빼시오
# 조건2. 히든 레이어 X

from sklearn.svm import SVC
# svm(Support Vector Machine)
# 서포트벡터 : 아웃풋 중에서 가장 경계선에 가까이 붙어있는 최전방의 데이터들
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# KNeighbors 최근접 이웃 / 데이터 별로 근접한 기준을 사용
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import accuracy_score

# 왜 데이터에서 충돌? 넘파이는 행렬 연산으로 레이어마다 가중치의 곱으로 계산
# 머신러닝에서는 가중치 계산 X(하긴 함). 데이터가 리스트로 가능
# 라벨 인코더를 사용하여 글씨도 그냥 수치로 바꿀수 있다
# 라벨 인코더에 데이터셋을 넣으면 사람 1, 고양이 2 이렇게 알아서 분류

#1. 데이터
x_data = np.array([[0,0], [1,0], [0,1], [1,1]])       # (4,2)
y_data = np.array([0,1,1,0])                          # (4, )
# print(x_data)
# [[0 0]
#  [1 0]
#  [0 1]
#  [1 1]]
# print(y_data)     # [0 1 1 0]

#2. 모델
model = Sequential()
model.add(Dense(1, input_shape=(2, )))
# model.add(Dense(9000))
# model.add(Dense(5000))
# model.add(Dense(300, activation='sigmoid'))
# model.add(Dense(200))
# model.add(Dense(1, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100, batch_size=1)

#4. 평가, 예측
x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_pred = model.predict(x_test)

y_test = np.array([0,1,1,0])
loss_acc = model.evaluate(x_test, y_test)
# score == 케라스의 model.evaluate(평가)
# acc = accuracy_score([0,1,1,0], y_pred)

print(x_test, "의 예측 결과: \n", y_pred)
# [[0 0]
#  [1 0]
#  [0 1]
#  [1 1]] 의 예측 결과:
#  [[ 0.05505804]
#  [-1.3277211 ]
#  [ 0.7627079 ]
#  [-0.62007123]]
print("loss, acc : ", loss_acc)
# loss, acc :  [3.938115119934082, 0.5]
# loss_acc를 리스트로 묶어주고 print하면 리스트 형태로 출력
