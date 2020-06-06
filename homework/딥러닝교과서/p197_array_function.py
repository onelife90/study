# 범용 함수 : ndarray 배열의 각요소에 대한 연산결과를 반환
# 요소별로 계산하므로 다차원 배열에도 사용
# 인수가 하나인 경우 = 절대값 반환 np.abs(), e의 거듭제곱 반환 np.exp(), 제곱근 반환 np.sqrt()
# 인수가 두개인 경우 = 합 np.add(), 차이 np.subtract(), 최대값 배열 반환 np.maximum()
import numpy as np
arr = np.array([4,-9,16,-4,20])
# arr의 각 요소를 절대값으로 하여 변수 arr_abs에 대입
arr_abs = np.abs(arr)       
print(arr_abs)      # [ 4  9 16  4 20]
# arr_abs의 각 요소의 e의 거듭제곱과 제곱근을 출력
print(np.exp(arr_abs))      # [5.45981500e+01 8.10308393e+03 8.88611052e+06 5.45981500e+01 4.85165195e+08]
print(np.sqrt(arr_abs))     # [2.         3.         4.         2.         4.47213595]

# 함수에 np. 붙이는 거 잊지말긔
