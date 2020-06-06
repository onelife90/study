# numpy 파이썬으로 벡터나 행렬 계산을 빠르게 하도록 특화된 라이브러리
# 라이브러리 : 외부에서 읽어 들이는 파이썬 코드 묶음
# 함수 덩어리>모듈>라이브러리

# numpy 고속 처리 경험
import  numpy as np
import time
from numpy.random import rand

# 행, 열 크기
N = 150
# 행렬 초기화
matA = np.array(rand(N,N))
matB = np.array(rand(N,N))
matC = np.array([[0]*N for  _ in range(N)])
# 파이썬 리스트 사용하여 계산
# 시작 시간 저장
start = time.time()
# for문을 사용하여 행렬 곱셈 실행
for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k]*matB[k][j]
print("파이썬 기능만으로 계산한 결과:  %.2f[sec]" % float(time.time() - start))

# numpy를 사용하여 계산
# 시작 시간 저장
start = time.time()
# numpy를 사용하여 행렬 곱셈 실행
matC = np.dot(matA, matB)
print("numpy 기능만으로 계산한 결과:  %.2f[sec]" % float(time.time() - start))
# 파이썬 기능만으로 계산한 결과:  2.24[sec]
# numpy 기능만으로 계산한 결과:  0.00[sec]

# numpy에는 배열을 고속으로 처리하는 ndarray클래스
# 또 다른 방법으로는 np.array() 함수 이용
# np.arange() 함수 이용. 일정한 간격으로 증감시칸 값의 요소 생성
# ndarray 클래스. 1차원 벡터, 2차원 행렬, 3차원 이상 텐서
# 1차원 클래스 array_1d = np.array([1,2,3,4,5,6,7,8])               # (8, )
# 2차원 클래스 array_2d = np.araay([1,2,3,4], [5,6,7,8])            # (2,4)
# 3차원 클래스 array_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])     # (2,2,2)
array_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(array_3d.shape)   # (2, 2, 2)

import numpy as np
storages = [23,3,4,23,10,12]
print(storages)     # [23, 3, 4, 23, 10, 12]
# ndarray 배열을 생성하여 변수 np_storages에 대입
np_storages = np.array(storages)
# 변수 np_storages의 자료형 출력
print(type(np_storages))        # <class 'numpy.ndarray'>

# 1차원 배열의 계산
import numpy as np
arr = np.array([2,5,3,4,8])
# arr+arr
print(arr+arr)      # [ 4 10  6  8 16]
# arr-arr
print(arr-arr)      # [0 0 0 0 0]
# arr**3
print(arr**3)       # [  8 125  27  64 512]
# 1/arr
print(1/arr)        # [0.5        0.2        0.33333333 0.25       0.125     ]
