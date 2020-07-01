import numpy as np
import matplotlib.pyplot as plt

# arange 0~10까지 0.1씩 증가시켜서 array형태로 반환해주는 함수
# numpy.arange([start, ] stop, [step, ]) == start에서 stop까지 step으로 array로 변환
x = np.arange(0, 10, 0.1)
print(x) # [x xx xxx xxxx]

# 여기서 급 궁금점. 리스트와 array(배열)은 다른가요?
# array(배열) : 배열은 인덱스라는 장점
# 리스트 : 빈틈없는 데이터의 적재. 그래서 콤마가 들어감
#ex)
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x)     [ 1  2  3  4  5  6  7  8  9 10]

y = np.sin(x)

plt.plot(x,y)

plt.show()
