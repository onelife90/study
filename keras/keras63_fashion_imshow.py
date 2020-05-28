import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# imshow를 이용하여 fashion_mnist 이미지 출력

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("x_train[0]: \n", x_train[0])
print("y_train[0]: \n", y_train)

print(x_train.shape)        # (60000, 28, 28)
print(x_test.shape)         # (10000, 28, 28)
print(y_train.shape)        # (60000,)
print(y_test.shape)         # (10000,)

plt.imshow(x_train[3200])
plt.show()
