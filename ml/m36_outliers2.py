# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수 구현
# 파일명 : m36_outliers2.py

import numpy as np

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사분위 : ", quartile_1)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
    # | == or 둘 중 하나 구간을 찾아서 np.where 넘파이 장소를 찾아서 return 해라

# a = np.array([[1,2,3,4,10000,6,7,5000,90,100],[10,20,30,40,500031,60,70,80,90,100]])
# a = a.transpose()
a = np.array([[1,2],[3,4],[500,600],[7,8],[9,10000],[10,20],[30,40],[500031,60],[70,80],[90,100]])
a = np.mat(a)
# a = np.matrix('1 2; 3 4; 500 600; 7 8; 9 10000; 10 20; 30 40; 500031 60; 70 80; 90 100')
print(a.shape)      # (10, 2)
b = outliers(a)
print("이상치의 위치 : ", b)
# 1사분위 :  7.75
# 3사분위 :  92.5
# 이상치의 위치 :  (array([2, 2, 4, 7], dtype=int64), array([0, 1, 1, 0], dtype=int64))
