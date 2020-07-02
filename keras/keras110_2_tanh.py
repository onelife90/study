# tanh라는 활성화 함수를 만들어보자

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
y = np.tanh(x)
# 탄젠트는 넘파이에서 제공하기 때문에 함수 만들지 않아도 됨

plt.plot(x,y)
plt.grid()
plt.show()
# y값은 -1과 1 사이의 값으로 수렴
