# relu라는 함수를 살펴보자

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5,5,0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()
# y값은 0이하면 무조건 0으로 수렴, 0 이상이면 선형으로 증가
