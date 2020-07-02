# leakyrelu 함수를 살펴보자

import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x):
    return np.maximum(0.15*x, x)
    # 0.15 자리가 알파값인데 튜닝 가능
    # import 해주고 나서 튜닝하려면 복잡해짐

x = np.arange(-5,5,0.1)
y = leaky_relu(x)

plt.plot(x,y)
plt.grid()
plt.show()
