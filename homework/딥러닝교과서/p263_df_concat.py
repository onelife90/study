# df을 일정 방향으로 붙이는 작업 : 연결
# df의 특정 key를 참조하여 연결하는 조작 : 결합
# pd.concat("DataFrame리스트", axis=0) : 리스트 선두부터 세로로 연결
# 세로 방향으로 연결 시 동일한 컬럼으로 연결, 가로 방향 연결 시 동일한 인덱스로 연결
# 그대로 연결하므로 컬럼에 중복된 값이 생길 수 있음

#1) 인덱스나 컬럼이 일치하는 df간의 연결
import  numpy as np
import pandas as pd
# 지정한 인덱스와 컬럼을 가진 df을 난수로 생성하는 함수
def make_random_df(index,columns,seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1,101), len(index))
    df.index = index
    return df
# 인덱스와 컬럼이 일치하는 df를 만듬
columns = ["apple", "orange","banana"]
df_data1 = make_random_df(range(1,5), columns, 0)
df_data2 = make_random_df(range(1,5), columns, 1)
# df_data1과 df_data2를 세로로 연결하여 df1에 대입
df1 = pd.concat([df_data1,df_data2], axis=0)
# df_data1과 df_data2를 가로로 연결하여 df2에 대입
df2 = pd.concat([df_data1, df_data2], axis=1)
print(df1)
#    apple  orange  banana
# 1     45      68      37
# 2     48      10      88
# 3     65      84      71
# 4     68      22      89
# 1     38      76      17
# 2     13       6       2
# 3     73      80      77
# 4     10      65      72
print(df2)
#    apple  orange  banana  apple  orange  banana
# 1     45      68      37     38      76      17
# 2     48      10      88     13       6       2
# 3     65      84      71     73      80      77
# 4     68      22      89     10      65      72

#2) 인덱스와 컬럼이 일치하지 않는 df간의 연결
# 공통의 인덱스나 컬럼이 아닌 행과 열에는 nan 생성
import  numpy as np
import pandas as pd
def make_random_df(index,columns,seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1,101), len(index))
    df.index = index
    return df
columns1 = ["apple", "orange","banana"]
columns2 = ["orange", "kiwi","banana"]
# 인덱스가 1,2,3,4고 컬럼이 columns1인 df을 생성
df_data1 = make_random_df(range(1,5), columns1,0)
# 인덱스가 1,3,5,7이고 컬럼이 columns2인 df 생성
df_data2 = make_random_df(np.arange(1,8,2), columns2, 1)
# df_data1과 df_data2를 세로로 연결하여 df1에 대입
df1 = pd.concat([df_data1, df_data2], axis=0)
# df_data1과 df_data2를 가로로 연결하여 df2에 대입
df2 = pd.concat([df_data1, df_data2], axis=1)
print(df1)
#    apple  orange  banana  kiwi
# 1   45.0      68      37   NaN
# 2   48.0      10      88   NaN
# 3   65.0      84      71   NaN
# 4   68.0      22      89   NaN
# 1    NaN      38      17  76.0
# 3    NaN      13       2   6.0
# 5    NaN      73      77  80.0
# 7    NaN      10      72  65.0
print(df2)
#    apple  orange  banana  orange  kiwi  banana
# 1   45.0    68.0    37.0    38.0  76.0    17.0
# 2   48.0    10.0    88.0     NaN   NaN     NaN
# 3   65.0    84.0    71.0    13.0   6.0     2.0
# 4   68.0    22.0    89.0     NaN   NaN     NaN
# 5    NaN     NaN     NaN    73.0  80.0    77.0
# 7    NaN     NaN     NaN    10.0  65.0    72.0
