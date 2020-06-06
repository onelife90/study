# np.random 모듈로 난수 생성
# 0~1 난수 생성 np.random.rand(), x 이상 y미만의 정수를 z개 생성 np.random.randint(x,y,z), 가우스 분포를 따르는 난수 생성 np.random.normal()
# np.random.rand()함수에는 ()에 넣은 정수 횟수만큼 난수 생성
# np.random.randint(x,y,z)는 x이상 y미만의 정수를 생성. z에는 (2,3)의 인수를 넣을 수도 있는데 2X3 행렬을 생성
import  numpy as np
# np.random을 적지 않아도 randint()함수를 사용하게 import하세요
from numpy.random import randint
# arr1에 각 요소가 0이상 10 이하인 정수 행렬5X2를 대입
arr1 = randint(0,11,(5,2))
print(arr1)
# [[ 1  3]
#  [ 4  5]
#  [ 4  0]
#  [10  5]
#  [ 1  3]]
# arr2에 0이상 1미만의 난수 3개 생성해서 대입
arr2 = np.random.rand(3)
print(arr2)     # [0.57057829 0.82258161 0.21804554]
