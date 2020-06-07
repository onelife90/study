# 데이터 양이 방대할 떄 화면에 전부 출력하기 어려움. 특정 행을 얻을 수 있음
# 첫 5행만 빈환 df.head(), 끝 5행만 반환 df.tail()
# 인수로 정수값을 지정하면 처음이나 끝에서부터 특정 행까지의 df를 얻음
import  numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange","banana","strawberry","kiwi"]
df = pd.DataFrame()

for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# df의 첫 3헹을 취득하여 df_head에 대입
df_head = df.head(3)
# df의 끝 3행을 취득하여 df_tail에 대입
df_tail = df.tail(3)
print(df_head)
#    apple  orange  banana  strawberry  kiwi
# 1      6       8       6           3    10
# 2      1       7      10           4    10
# 3      4       9       9           9     1
print(df_tail)
#     apple  orange  banana  strawberry  kiwi
# 8       6       8       4           8     8
# 9       3       9       6           1     3
# 10      5       2       1           2     1
