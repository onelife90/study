# ndarray의 슬라이스는 배열의 복사본이 아닌 원래 배열을 슬라이스
import numpy as np
# 파이썬의 리스트에 슬라이스를 이용한 경우
arr_List = [x for x in range(10)]
print("리스트형 데이터입니다")
print("arr_List: ", arr_List)
print()
# 리스트형 데이터입니다
# arr_List:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

arr_List_copy = arr_List[:]
arr_List_copy[0]=100
print("리스트의 슬라이스는 복사본이 생성되므로 arr_List에는 arr_List_copy의 변경이 반영되지 않습니다")
print("arr_List: ", arr_List)
# 리스트의 슬라이스는 복사본이 생성되므로 arr_List에는 arr_List_copy의 변경이 반영되지 않습니다
# arr_List:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# numpy의 ndarray에 슬라이스를 이용한 경우
arr_np = np.arange(10)
print("numpy의 ndarray 데이터입니다")
print("arr_np: ", arr_np)
print()
# numpy의 ndarray 데이터입니다
# arr_np:  [0 1 2 3 4 5 6 7 8 9]
arr_np_view = arr_np[:]
arr_np_view[0] = 100
print("numpy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로 arr_np_view를 변경하면 arr+np에 반영됩니다")
print("arr_np: ", arr_np)
print()
# numpy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로 arr_np_view를 변경하면 arr+np에 반영됩니다
# arr_np:  [100   1   2   3   4   5   6   7   8   9]

# numpy의 ndarray에서 copy()를 사용한 경우
arr_np = np.arange(10)
print("numpy의 ndarray에서 copy()를 사용한 경우 입니다")
print("arr_np: ", arr_np)
print()

arr_np_copy = arr_np[:].copy()
arr_np_copy[0] = 100
print("copy()를 사용하면 복사본이 생성되기 때문에 arr_np_copy는 arr_np에 영향을 미치지 않습니다")
print("arr_np: ", arr_np)
# copy()를 사용하면 복사본이 생성되기 때문에 arr_np_copy는 arr_np에 영향을 미치지 않습니다
# arr_np:  [0 1 2 3 4 5 6 7 8 9]
