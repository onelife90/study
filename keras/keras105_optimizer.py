# 현재 loss를 줄이는 작업
# 딥러닝에서 loss는 model.compile에 존재

#1. 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

#3. 컴파일, 훈련
# 최적화 함수를 import

# RMSprop : 학습률을 제외한 모든 인자의 기본값을 사용하는 것이 권장
# SGD : 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 옵티마이저
# Adadelta : Adagrad 확장버전
# Adagrad : 모델 파라미터별 학습률을 사용하는 옵티마이저
# Nadam : 네스테로프(누적된 과거 그래디언트가 지향하고 있는 어떤 방향을 현재 그라디언트에 보정하려는 방식) Adam 옵티마이저

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
# opti = Adam(lr=0.001)
# opti = RMSprop(lr=0.001)    # 0.05103191360831261, [[3.3807342]]
# opti = SGD(lr=0.001)        # 0.06409601867198944, [[3.3715045]]
# opti = Adadelta(lr=0.001)   # 7.475847244262695, [[0.00545527]]
# opti = Adagrad(lr=0.001)    # 5.187265872955322, [[0.5691264]]    
opti = Nadam(lr=0.001)      # 0.7942038774490356, [[2.305936]]

model.compile(loss='mse', optimizer=opti, metrics=['mse'])
model.fit(x,y,epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss: ", loss)

pred1 = model.predict([3.5])
print(pred1)
