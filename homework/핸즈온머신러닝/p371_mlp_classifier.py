from keras.datasets import  fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import plot_model
import pandas as pd
import matplotlib.pyplot as plt


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)    # (60000, 28, 28)
print(x_train.dtype)    # uint8

# 경사 하강법으로 신경망을 훈련하기 때문에 입력 특성의 스케일을 조정
# 간편하게 255로 나누어 0~1 범위로 조정
x_val, x_train = x_train[:5000]/255.0, x_train[5000:]/255.0
y_val, y_train = y_train[:5000], y_train[5000:]
x_test = x_test/255.0

# 레이블에 해당하는 아이템을 나타내기 위해 클래스 이름의 리스트 작성
class_name = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Sneaker","Bag","Ankle boot"]

#2. 모델 구성
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

hide = model.layers[1]
print(hide.name)    # dense_1
# hide.get_weights()

# model.summary()
# Dense층은 대칭성을 깨드리기 위해 연결 가중치를 무작위로 초기화. 편향은 0으로 초기화
# 다른 초기화 방법 kernel_initializer(커널은 연결 가중치 행렬의 또 다른 이름), bias_initializer

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc'])
# 만약 샘플마다 클래스별 타깃 확률을 가지고 있다면(즉, 원핫벡터라면) 'categorical_crossentropy' 손실 함수 사용
# 옵티마이저에 'sgd'를 지정하면 확률적 경사 하강법을 사용하여 모델을 훈련한다는 의미. 디폴트 lr=0.01

hist = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))
# 어떤 클래스는 많이 등장하고 다른 클래스는 조금 등장하여 훈련 세트가 편중되어 있다면(ex.wine데이터) fit() 메서드 호출시 class_weight 매개변수 지정 하는 것이 좋음

#4. 평가, 예측
model.evaluate(x_test, y_test)
x_pred = x_test[:3]
y_pred = model.predict(x_pred)

print("y_pred.proba: \n", y_pred.round(2))
# y_pred.proba: 
#  [[0.   0.   0.   0.   0.   0.   0.   0.02 0.   0.97]
#  [0.   0.   0.99 0.   0.01 0.   0.   0.   0.   0.  ]
#  [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]]


#5. 시각화
pd.DataFrame(hist.history).plot()
plt.figure(figsize=(8,5))
plt.grid()
plt.gca().set_ylim(0,1) # y축 범위를 [0~1] 사이로 설정
plt.legend()
plt.show()
