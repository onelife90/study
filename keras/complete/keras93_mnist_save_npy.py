import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
# mnist 손글씨로 된 7만장의 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# save 할 때는 arr(배열 형태로 저장, 변수명)
np.save('./data/mnist_train_x.npy', arr=x_train)     
np.save('./data/mnist_test_x.npy', arr=x_test)
np.save('./data/mnist_train_y.npy', arr=y_train)
np.save('./data/mnist_test_y.npy', arr=y_test)

# 데이터 전처리 1. one-hot 인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255
