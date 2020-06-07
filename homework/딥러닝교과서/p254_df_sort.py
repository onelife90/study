# df.sort_values(by="컬럼 또는 컬럼리스트", ascending=True)를 지정하여 열의 값을 오름차순 정렬
# ascending=False 내림차순
# 컬럼 리스트에서 순서가 빠른 열이 우선 적용
import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# df를 "apple", "orange", "banana", "strawberry", "kiwi" 순으로 오름차순 정렬
df = df.sort_values(by=columns)
print(df)
#     apple  orange  banana  strawberry  kiwifruit
# 2       1       7      10           4         10
# 9       3       9       6           1          3
# 7       4       8       1           4          3
# 3       4       9       9           9          1
# 4       4       9      10           2          5
# 10      5       2       1           2          1
# 8       6       8       4           8          8
# 1       6       8       6           3         10
# 5       8       2       5           4          8
# 6      10       7       4           4          4
