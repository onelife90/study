# 번호 참조 (인덱스와 컬럼 번호)로 참조 ==> iloc
# df.iloc["행번호 리스트", "열번호리스트"]

import numpy as np
import pandas as pd

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwi"]

# DataFrame을 생성하고 열 추가
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)

# iloc[]을 사용하여 df의 2~5행과 "banana", "kiwi"의 2열을 포함하여 df에 대입
df = df.iloc[1:5, [2,4]]
# 방법1. 슬라이싱 : 2~5행이니 인덱스행으로 따지면 0:5가 되며, 이 때 리스트형태로 묶어주지 않고 바로 씀
# 방법2. range : range(1,5)==인덱스 1번째 행부터 인덱스 4번째 행까지 즉, 2,3,4,5행을 사용. 이 때도 리스트 형태로 묶어주지 X
print(df)
