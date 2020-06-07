# df도 Series와 마찬가지로 bool형의 시퀀스를 지정하여 True만 추출하는 필터링 가능
# df.loc[df["컬럼"]조건식]으로 지정
import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwi"]
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# 필터링을 사용하여 df의 "apple"열이 5이상, "kiwi"열이 5이상의 값을 가진 행을 포함한 DataFrame을 df에 대입
df = df.loc[df["apple"]>=5]
df = df.loc[df["kiwi"]>=5]
print(df)
#    apple  orange  banana  strawberry  kiwi
# 1      6       8       6           3    10
# 5      8       2       5           4     8
# 8      6       8       4           8     8
