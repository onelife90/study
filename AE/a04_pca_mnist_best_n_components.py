# 컬럼을 압축해보자

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()
# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255
# print(f'x_train.shape: {x_train.shape}')        # (60000, 784)
# print(f'x_test.shape: {x_test.shape}')         # (10000, 784)

#1-2. X데이터 train+test 합치기
X = np.append(x_train, x_test, axis=0)
print(f'X.shape: {X.shape}')    # X.shape: (70000, 784)

#1-3. 차원을 축소하지 않고 PCA 계산 후 최적의 n_components 구하기
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f'cumsum: {cumsum}')

best_n_components = np.argmax(cumsum>=0.95)+1
print(f'best_n_components: {best_n_components}')
# n_components: 154
