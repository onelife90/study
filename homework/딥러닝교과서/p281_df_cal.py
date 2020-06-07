# pandas와 numpy는 상호호환이 좋아서 유연한 데이터 전달 가능
# 판다스는 브로드캐스트를 지원하므로 판다스간의 계산 혹은 판다스와 정수간의 계산을 사칙연산을 사용해서 유연하게 처리
import  numpy as np
import pandas as pd
import math
np.random.seed(0)
columns = ["apple", "orange","banana","strawberry","kiwi"]
df = pd.DataFrame()

for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# df의 각 요소를 두배로 만들어 double_df에 대입
double_df = df*2
# df의 각 요소를 제곱하여 square_df에 대입
square_df = df**2
# df의 각 요소를 제곱근 계산하여 sqrt_df에 대입
sqrt_df = np.sqrt(df)
print(double_df)
#     apple  orange  banana  strawberry  kiwi
# 1      12      16      12           6    20
# 2       2      14      20           8    20
# 3       8      18      18          18     2
# 4       8      18      20           4    10
# 5      16       4      10           8    16
# 6      20      14       8           8     8
# 7       8      16       2           8     6
# 8      12      16       8          16    16
# 9       6      18      12           2     6
# 10     10       4       2           4     2
print(square_df)
#     apple  orange  banana  strawberry  kiwi
# 1      36      64      36           9   100
# 2       1      49     100          16   100
# 3      16      81      81          81     1
# 4      16      81     100           4    25
# 5      64       4      25          16    64
# 6     100      49      16          16    16
# 7      16      64       1          16     9
# 8      36      64      16          64    64
# 9       9      81      36           1     9
# 10     25       4       1           4     1
print(sqrt_df)
#        apple    orange    banana  strawberry      kiwi
# 1   2.449490  2.828427  2.449490    1.732051  3.162278
# 2   1.000000  2.645751  3.162278    2.000000  3.162278
# 3   2.000000  3.000000  3.000000    3.000000  1.000000
# 4   2.000000  3.000000  3.162278    1.414214  2.236068
# 5   2.828427  1.414214  2.236068    2.000000  2.828427
