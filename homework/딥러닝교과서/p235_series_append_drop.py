# series에 요소를 추가하려면 해당 요소도 series형이어야 함
# 추가할 요소를 Series형으로 변환한 뒤 Series형의 appen()로 추가
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
series = pd.Series(data, index=index)
# 인덱스가 "pineapple"이고 데이터가 12인 요소를 series에 추가
pineapple = pd.Series([12], index=["pineapple"])
series = series.append(pineapple)
print(series)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwi           3
# pineapple     12
# dtype: int64
# 새로운 Series 지정시, ([데이터],[인덱스])로 추가해야함

# 요소 삭제도 가능 Series.drop("인덱스")로 해당 인덱스 위치의 요소 삭제
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
series = pd.Series(data, index=index)
# 인덱스가 strawberry인 요소를 삭제하고 series에 대입
series = series.drop("strawberry")
print(series)
# apple     10
# orange     5
# banana     8
# kiwi       3
# dtype: int64
