# 2차원 배열은 행렬에 해당 np.array(리스트, 리스트)로 표기
# ndarray배열.shape로 각 차원이 요소 수 반환
# ndarray배열.reshape(a,b)로 지정한 인수와 같은 모양의 행렬로 변환
import numpy as np
# arr에 2차원 배열을 대입
arr = np.array([[1,2,3,4],[5,6,7,8]])
print(arr)
# [[1 2 3 4]
#  [5 6 7 8]]
# arr 행렬의 각 차원의 요소 수 출력
print(arr.shape)  # (2,4)
# arr을 4행 2열의 행렬로 반환
print(arr.reshape(4,2))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# shape 계산법
#1) 가장 작은 [] 안에 요소가 몇 개 인지 확인
#2) 그 작은 []이 괄호로 몇 개 인지 확인
#3) 마지막으로 크게 묶인 []가 몇 개 인지 확인
