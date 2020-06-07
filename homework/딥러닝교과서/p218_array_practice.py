import  numpy as np
np.random.seed(100)
# 각 요소가 0~30인 정수 행렬(5X3)을 변수 arr에 대입
arr = np.random.randint(0,31,(5,3))
print(arr)
# [[ 8 24  3]
#  [ 7 23 15]
#  [16 10 20]
#  [ 2 21  2]
#  [ 2 14  2]]
# arr을 전치
arr = arr.T
print(arr)
# [[ 8  7 16  2  2]
#  [24 23 10 21 14]
#  [ 3 15 20  2  2]]
# arr의 2,3,4열만 추출한 행렬(3X3)을 arr1에 대입
arr1 = arr[:, 1:4]
print(arr1)
# [[ 7 16  2]
#  [23 10 21]
#  [15 20  2]]
# arr1 행을 정렬
arr1.sort(1)
print(arr1)
# [[ 2  7 16]
#  [10 21 23]
#  [ 2 15 20]]
# 각 열의 평균 출력
print(arr1.mean(axis=0))
# [ 4.66666667 14.33333333 19.66666667]

# 난수 초기화
np.random.seed(0)
# 가로세로 크기를 전달하면 해당 크기의 이미지를 난수로 채워서 생성하는 함수
def make_image(m,n):
    # nXm 행렬의 각 성분을 0~5의 난수로 채우시오
    image = np.random.randint(0,6,(n,m))
    return image
#전달된 행렬의 일부 변경 함수
def change_little(matrix):
    # 전달받은 행렬의 형태를 취득하여 shape에 대입
    shape = matrix.shape
    # 행렬의 각 성분에 대해 변경 여부를 무작위로 결정한 다음 0~5 사이의 정수로 임의 교체
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0,2)==1:
                matrix[i][j] = np.random.randint(0,6,1)
    return matrix
# 임의의 이미지 만들기
image1 = make_image(3,3)
print(image1)
# [[4 5 0]
#  [3 3 3]
#  [1 3 5]]
print()
# 임의의 변경사항 저장
image2 = change_little(np.copy(image1))
print(image2)
# [[4 5 0]
#  [3 3 3]
#  [0 5 5]]
print()
# image1과 image2의 차이를 계산하여 image3에 대입
image3 = image2 - image1
print(image3)
# [[ 0  0  0]
#  [ 0  0  0]
#  [-1  2  0]]
print()
# image3의 각 성분을 절대값으로 한 행렬을 image3에 다시 대입
image3 = np.abs(image3)
print(image3)
# [[0 0 0]
#  [0 0 0]
#  [1 2 0]]
