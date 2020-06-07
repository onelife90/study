import pandas as pd
import numpy as np
index = ["growth","mission","ishikawa","pro"]
data = [50,7,26,1]
# Series 작성
series = pd.Series(data,index=index)
# 인덱스를 알파벳순으로 정렬한 series를 aidemy에 대입
aidemy =  series.sort_index()
# 인덱스가 "tutor"고 데이터가 30인 요소 series에 추가
aidemy1 = pd.Series([30], index=["tutor"])
aidemy2 = series.append(aidemy1)
print(aidemy)
# growth      50
# ishikawa    26
# mission      7
# pro          1
# dtype: int64
print()
print(aidemy2)
# growth      50
# mission      7
# ishikawa    26
# pro          1
# tutor       30
# dtype: int64
# DataFrame을 생성하고 열 추가
df = pd.DataFrame()
for index in index:
    df[index] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
# loc[]를 사용하여 df의 2~5행과 "ishikawa"의 열을 포함한 DataFrame을 aidemy3에 대입
aidemy3 = df.loc[range(2,6),["ishikawa"]]
print()
print(aidemy3)
#    ishikawa
# 2         8
# 3         1
# 4         2
# 5         6
