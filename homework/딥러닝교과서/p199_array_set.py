# 집합 함수 : 수학의 집합 연산을 수행. 1차원 배열만을 대상
# 배열 요소에서 중복을 제거하고 정렬한 결과 반환 np.unique(), 합집합 np.union1d(), 교집합 np.intersect1d(), 차집합 np.setdified1d()
import numpy as np
arr1 = [2,5,7,9,5,2]
arr2 = [2,5,8,3,1]
# np.unique() 사용해서 new_arr1에 대입
new_arr1 = np.unique(arr1)
print(new_arr1)     # [2 5 7 9]
# 합집합
print(np.union1d(new_arr1, arr2))       # [1 2 3 5 7 8 9]
# 교집합
print(np.intersect1d(new_arr1, arr2))   # [2 5]
# 차집합
print(np.setdiff1d(new_arr1, arr2))     # [7 9]
