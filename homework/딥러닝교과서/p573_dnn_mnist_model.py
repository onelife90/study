# 모델 생성
# 먼저 케라스로 모델을 관리하는 인스턴스를 만들고, add() 메서드로 한 층씩 정의
# model = Sequential ==> 인스턴스 형성
# 각 전결합층의 출력은 다음처럼 활성화 함수 적용
# model.add(Activation("sigmoid"))
# 마지막으로 컴파일 메서드 compile()을 이용하여 어떠한 학습을 실시할지 설정 모델 생성 종료

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)[:6000]
x_test = x_test.reshape(x_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

#2. 모델구성
model = Sequential()
# 입력 유닛 수는 784개며, 첫 번째 전결합층의 출력 유닛 수는 256
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
# 두번째 전결합층의 출력 유닛 수는 10
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Dense(10))
model.add(Activation("softmax"))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

# 모델 구조 출력
plot_model(model, "model125.png", show_layer_names=False)
# 모델 구조 시각화
image = plt.imread("model125.png")
plt.figure(dpi=150)  # dpi=해상도
plt.imshow(image)
plt.show()
