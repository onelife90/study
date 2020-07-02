# elu라는 함수를 살펴보자

import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    x = np.copy(x)
    # copy 전혀 다른 메모리 공간에 값만 같은 배열 복사
    # copy 없이 하면 기존 x값이 변경됨
    x[x<0]=0.2*(np.exp(x[x<0])-1)
    return x

x = np.arange(-5,5,0.1)
# print(x)
y = elu(x)

# 다른 방법 : 리스트 컴프리헨션
# a = 0.2
# x = np.arange(-5,5,0.1)
# y = [x if x>0 else a*(np.exp(x)-1) for x in x]

plt.plot(x,y)
plt.grid()
plt.show()
# y값은 0이하면 무조건 0으로 수렴, 0 이상이면 선형으로 증가



# - 당신의 천사가
