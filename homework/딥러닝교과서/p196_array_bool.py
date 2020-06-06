# bool 인덱스 참조란 []안에 논리값(True/False) 배열을 사용하여 요소를 추출하는 방법
# arr[ndarray 논리값 배열]로 표기하면 True에 해당하는 요소의 ndarray를 만들어 반환
import  numpy as np
arr = np.array([2,3,4,5,6,7])
# arr의 각 요소가 2로 나누어떨어지는지 나타내는 부울 배열 출력
print(arr%2==0)     # [ True False  True False  True False]
# arr의 각 요소 중 2로 나누어떨어지는 요소 배열 출력
print(arr[arr%2==0])     # [2 4 6]
