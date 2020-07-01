# 이상치 제거
# 1) 삭제 : 범위를 정해두고 범위 밖 수치를 제거하는 방법
# 2) nan 처리 : 보간법 사용 가능
#ex) 1, 2, 3, 100, 500, 6, 7 이 컬럼들이 화장실 갯수라 치면 100과 500은 이상치
# 판단은 본인이

# 감성적 데이터 분석할 때 자주 사용하는 IQR
# 이상치 : 범위 밖 튀어나오는 데이터
# 전체 데이터 4분위
# 이상치는 25% 전이나 75% 밖의 범위에 있음
# 25% * 1.5, 75% * 1.5한 범위에 이상치가 있다고 가정

import numpy as np

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사분위 : ", quartile_1)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

a = np.array([1,2,3,4,10000,6,7,5000,90,100])
b = outliers(a)
print("이상치의 위치 : ", b)

# 데이콘1 컬럼별로 이상치가 나올 수 있게 for문 써서 만들어라
