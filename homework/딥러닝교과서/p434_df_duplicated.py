# 중복 데이터 삭제
# duplicated() 메서드를 사용하면 중복된 행을 True로 표시 출력 형태는 Series형
# 삭제하려면? drop_duplicates() 메서드 사용하여 중복된 데이터가 삭제된 후의 데이터 출력
# 모든 요소가 같아야 삭제가 됨
import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1": [1,1,2,3,4,4,6,6,7,7,7,8,9,9],
                        "col2": ["a","b","b","b","c","c","b","b","d","d","c","b","c","c"]})
overap = dupli_data.duplicated()
print(overap)
# 0     False
# 1     False
# 2     False
# 3     False
# 4     False
# 5      True
# 6     False
# 7      True
# 8     False
# 9      True
# 10    False
# 11    False
# 12    False
# 13     True
# dtype: bool
# 중복된 행을 True로 표시

drop = dupli_data.drop_duplicates()
print(drop)
#     col1 col2
# 0      1    a
# 1      1    b
# 2      2    b
# 3      3    b
# 4      4    c
# 6      6    b
# 8      7    d
# 10     7    c
# 11     8    b
# 12     9    c
