# 시계열 분석에서 행간 차이를 구하는 작업이 자주 이용
# df.diff("행간격 또는 열 간격", axis="방향")
# 첫 번째 인수가 양수면 이전 행과의 차이, 음수면 다음 행과의 차이
# axis=0이면 행의방향, axis=1이면 열의 방향
import  numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange","banana","strawberry","kiwi"]
df = pd.DataFrame()

for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# df의 각 행에 대해 2행 뒤와의 차이를 계산한 df를 df_diff에 대입
df_diff = df.diff(-2, axis=0)
print(df)
#     apple  orange  banana  strawberry  kiwi
# 1       6       8       6           3    10
# 2       1       7      10           4    10
# 3       4       9       9           9     1
# 4       4       9      10           2     5
# 5       8       2       5           4     8
# 6      10       7       4           4     4
# 7       4       8       1           4     3
# 8       6       8       4           8     8
# 9       3       9       6           1     3
# 10      5       2       1           2     1
print(df_diff)
#     apple  orange  banana  strawberry  kiwi
# 1     2.0    -1.0    -3.0        -6.0   9.0
# 2    -3.0    -2.0     0.0         2.0   5.0
# 3    -4.0     7.0     4.0         5.0  -7.0
# 4    -6.0     2.0     6.0        -2.0   1.0
# 5     4.0    -6.0     4.0         0.0   5.0
# 6     4.0    -1.0     0.0        -4.0  -4.0
# 7     1.0    -1.0    -5.0         3.0   0.0
# 8     1.0     6.0     3.0         6.0   7.0
# 9     NaN     NaN     NaN         NaN   NaN
# 10    NaN     NaN     NaN         NaN   NaN
# 9,10행의 값이 nan인 이유는 다음 행이 없기 떄문
