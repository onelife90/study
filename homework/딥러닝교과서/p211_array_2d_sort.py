# sort() 메서드로 정렬 가능
# 2차원 배열의 경우 0을 인수로 하면 열 단위 요소 정렬, 1을 인수로 하면 행 단위 요소 정렬
# np.sort() 함수로도 정렬. 정렬된 배열의 복사본을 반환. 원본 보존
# 머신러닝에 자주 사용되는 함수로 argsort() 메서드.  정렬된 배열의 본래 인덱스 반환
import  numpy as np
arr = np.array([[8,4,2],[3,5,1]])
# argsort() 메서드로 정렬하여 출력
print(arr.argsort())
# [[2 1 0]
#  [2 0 1]]
# np.sort() 함수로 정렬하여 출력
print(np.sort(arr))
# [[2 4 8]
#  [1 3 5]]
# sort() 메서드로 행을 정렬하여 출력
arr.sort(1)
print(arr)
# [[2 4 8]
#  [1 3 5]]
