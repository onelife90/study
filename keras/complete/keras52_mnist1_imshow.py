import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
# 0~255의 숫자가 컬럼으로 찍혀있다
# print("y_train: ", y_train[0])  # y_train:  5

# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])    # 랜덤색깔
# print(x_train[0].shape)   # (28, 28)
# plt.imshow(가로, 세로)==가로, 세로를 넣어주면 이미지를 출력
# plt.show()

