# 컬럼별로 데이터의 평균값, 최대값, 최소값 등의 통계 정보 집계
# df.describe() : 컬럼당 데이터수, 평균값, 표준편차, 최소값, 사분위수(25%,50%,75%), 최대값 정보를 포함하는 df를 반환
import  numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange","banana","strawberry","kiwi"]
df = pd.DataFrame()

for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# df의 통계 정보 중 mean, max, min을 꺼내서 df_des에 대입
df_des = df.describe().loc[["mean","max","min"]]
print(df_des)
#       apple  orange  banana  strawberry  kiwi
# mean    5.1     6.9     5.6         4.1   5.3
# max    10.0     9.0    10.0         9.0  10.0
# min     1.0     2.0     1.0         1.0   1.0
