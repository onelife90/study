import numpy as np
import matplotlib.pyplot as plt

# arange 0~10까지 0.1씩 증가시켜서 array형태로 반환해주는 함수
# numpy.arange([start, ] stop, [step, ]) == start에서 stop까지 step으로 array로 변환
x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x,y)

plt.show()
