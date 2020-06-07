# 브로드캐스트 : 크기가 다른 numpy 배열 간의 연산
# 크기가 작은 배열의 행과 열을 자동으로 큰 배열 쪽에 맞춤
# 행이 불일치 : 행이 적은 쪽이 많은 쪽의 수에 맞춰 부족한 부분을 기존 행에서 복사
import numpy as np
# 0에서 14 사이의 정숫값을 갖는 3X5의 ndarray 배열 x 생성
x = np.arange(15).reshape(3,5)
# 0에서 4 사이의 정수값을 갖는 1X5의 ndarray배열 y 생성
y = np.array([np.arange(5)])
# x에서 y를 빼세요
z = x-y
# x 출력
print(x)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]
