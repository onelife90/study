# 행렬 계산
# 두 행렬의 행렬곱을 반환하는 np.dot(a,b)와 노름을 반환하는 np.linalg.norm(a)
# 행렬곱 : 행렬에서 행벡터와 열벡터의 내적을 요소로 하는 행렬을 새로 생성
# 노름 norm은 벡터의 길이를 반환. 요소의 제곱값을 더해 루트를 씌운 것
# np.linalg.norm(a)
import  numpy as np
# arr 정의
arr = np.arange(9).reshape(3,3)
# arr과 arr의 행렬곱 출력
print(np.dot(arr,arr))
# [[ 15  18  21]
#  [ 42  54  66]
#  [ 69  90 111]]
# vec정의
vec = arr.reshape(9)
# vec의 노름 출력
print(np.linalg.norm(vec))      # 14.2828568570857
