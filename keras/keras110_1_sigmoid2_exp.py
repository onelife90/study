# 활성화 함수 sigmoid

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
    # np.exp 지수함수

x = np.arange(-5,5,0.1)
y = sigmoid(x)

print(x.shape, y.shape)
# (100,) (100,)

plt.plot(x,y)
plt.grid()
plt.show()
# y값이 0~1사이에 수렴
