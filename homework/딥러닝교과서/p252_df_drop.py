# df.drop()으로 인덱스 또는 컬럼을 지정하여 행이나 열 삭제 가능
# 열을 삭제할 시 두번째 인수로 axis = 1을 전달
import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# drop()을 이용하여 df에서 홀수 인덱스가 붙은 행만 남겨 df에 대입
df = df.drop(np.arange(2,11,2))
# drop()을 이용하여 df의 열 "strawberry"를 삭제하고 df에 대입
df = df.drop("strawberry", axis=1)
print(df)
#    apple  orange  banana  kiwifruit
# 1      6       8       6         10
# 3      4       9       9          1
# 5      8       2       5          8
# 7      4       8       1          3
# 9      3       9       6          3

# 홀수 인덱스만 남기려고 %2==0 이라고만 썼으나 eroor
# 실행시키려면 if문 만들어야할듯
