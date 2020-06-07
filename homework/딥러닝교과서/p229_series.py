# series는 1차원 배열
# pd.Series(딕셔너리형의 리스트)
# 데이터에 관련된 인덱스를 지정해도 Series 생성. pd.Series(데이터 배열, index=인덱스 배열)
# 인덱스를 지정하지 않으면 0부터 순서대로 정수 인덱스 붙음
# dtype:int64 출력 dtype=data type 데이터의 자료형
import pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
# index와 data를 포함한 Series를 만들어 series에 대입
series = pd.Series(data, index=index)
print(series)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwi           3
