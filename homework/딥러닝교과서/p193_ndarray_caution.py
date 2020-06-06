# ndarray 배열을 다른 변수에 그대로 대입하여 변수 값을 변경하면 원래 ndarray의 배열도 변경
# ndarray를 복사하여 두개의 변수를 별도로 만들고 싶을때 copy() 메서드 사용

import numpy as np
arr1 = np.array([1,2,3,4,5])
print(arr1)     # [1 2 3 4 5]
# ndarray를 그대로 arr2 변수에 대입한 경우
arr2 = arr1
arr2[0] = 100
# arr2 변수를 변경하면 원래 변수도 영향
print(arr1)     # [100   2   3   4   5]
# ndarray를 copy()를 사용하여 arr2 변수에 대입한 경우
arr1 = np.array([1,2,3,4,5])
arr2 = arr1.copy()
arr2[0] = 100
print(arr1)     # [1 2 3 4 5]
