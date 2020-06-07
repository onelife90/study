# df끼리 연결하면 라벨이 중복되는 경우
# pd.concat()에 keys를 추가하여 라벨 중복 방지
# 연결한 뒤의 df는 복수라벨이 사용된 MultiIndex가 됨
import  numpy as np
import pandas as pd
def make_random_df(index,columns,seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1,101), len(index))
    df.index = index
    return df
columns = ["apple", "orange","banana"]
df_data1 = make_random_df(range(1,5), columns,0)
df_data2 = make_random_df(np.arange(1,8,2), columns, 1)
# df_data1과 df_data2를 가로로 연결하고 keys로 X와 Y를 지정하여 MultiIndex로 만든 뒤 df에 대입
df = pd.concat([df_data1, df_data2], axis=1, keys=["X","Y"])
# df의 Y라벨 banana를 Y_banana에 대입
Y_banana = df["Y", "banana"]
print(df)
#       X                   Y
#   apple orange banana apple orange banana
# 1  45.0   68.0   37.0  38.0   76.0   17.0
# 2  48.0   10.0   88.0   NaN    NaN    NaN
# 3  65.0   84.0   71.0  13.0    6.0    2.0
# 4  68.0   22.0   89.0   NaN    NaN    NaN
# 5   NaN    NaN    NaN  73.0   80.0   77.0
# 7   NaN    NaN    NaN  10.0   65.0   72.0
print()
print(Y_banana)
# 1    17.0
# 2     NaN
# 3     2.0
# 4     NaN
# 5    77.0
# 7    72.0
# Name: (Y, banana), dtype: float64
