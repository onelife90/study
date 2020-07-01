# wine의 y값이 편중되어 있는 문제!

import pandas as pd
import matplotlib.pyplot as plt

#1-1. 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0, index_col=None)

#1-2. 데이터 안의 계수를 그룹화
count_data = wine.groupby('quality')['quality'].count()
# groupby()연산자를 사용하여 그룹 별로 나누고, 각 그룹별로 집계함수를 적용, 그룹별 집계 결과를 하나로 합치는 단계
# split -> apply function -> combine

# 숫자가 몇개인지 묶어서 count_data라는 변수명에 대입
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64
# 머신도 5와 6으로 분류를 하기 때문에 낮은 acc

count_data.plot()
plt.show()

# 이 문제를 어떻게 해결할까?
