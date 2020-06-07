# Series의 요소를 참조할 시, 1) 번호 지정 2) 인덱스 값 지정
# 인덱스 값을 지정하는 경우 원하는 요소의 인덱스 값을 하나의 리스트로 정리한 뒤 참조 가능
import pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
series = pd.Series(data, index=index)
# 인덱스 참조를 사용하여 series의 2~4번째에 있는 세 요소를 추출하여 items1에 대입
items1 = series[1:4]
# 인덱스 값을 지정하는 방법으로 "apple","banana","kiwi"의 인덱스를 가진 요소를 추출하여 items2에 대입
items2 = series[["apple","banana","kiwi"]]
print(items1)
# orange         5
# banana         8
# strawberry    12
# dtype: int64
print()
print(items2)
# apple     10
# banana     8
# kiwi       3
# dtype: int64
