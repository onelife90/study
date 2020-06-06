# numpy도 인덱스 참조와 슬라이스 가능
# 슬라이스 값 변경 'arr[start:end]=변경하려는 값'
import numpy as np
arr = np.arange(10)     # [0 1 2 3 4 5 6 7 8 9]
print(arr)
# 변수 arr의 요소 중에서 3,4,5만 출력
print(arr[3:6])         # [3 4 5]
# 변수 arr의 요소 중에서 3,4,5를 24로 변경
arr[3:6] = 24
print(arr)               # [ 0  1  2 24 24 24  6  7  8  9]
