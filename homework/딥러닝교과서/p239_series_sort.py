# 인덱스 정렬series.sort_index() series.sort_values() 데이터 정렬 방법
# 특별히 인수를 지정하지 않으면 오름차순 정렬. ascending=False은 내림차순
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
series = pd.Series(data, index=index)
# series의 인덱스를 알파벳순으로 정렬해서 itmes1에 대입
items1 = series.sort_index()
# series의 데이터 값을 오름차순으로 정렬해서 items2에 대입
items2 = series.sort_values()
print(items1)
# apple         10
# banana         8
# kiwi           3
# orange         5
# strawberry    12
# dtype: int64
print()
print(items2)
# kiwi           3
# orange         5
# banana         8
# apple         10
# strawberry    12
