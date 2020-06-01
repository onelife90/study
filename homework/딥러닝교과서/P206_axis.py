import numpy as np

arr = np.array([[1,2,3], [4,5,12], [15,20,22]])

# arr 행의 합을 구하여 제시한 1차원 배열을 반환
print(arr.sum(axis=1))
