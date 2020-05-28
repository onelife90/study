# cifar10 : 10가지 이미지를 찾는 데이터

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

# 데이터 구조
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train[0]: \n", x_train[0])
print("y_train[0]: \n", y_train)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()
