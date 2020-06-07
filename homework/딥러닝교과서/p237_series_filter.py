# 필터링 : Series형 데이터에서 조건과 일치하는 요소를 꺼낼때 사용
# pandas에서는 bool형의 시퀀스를 지정해서 True인것만 추출 가능
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
series = pd.Series(data, index=index)
# series 요소 중에서 5 이상 10 미만의 요소를 포함하는 Seires를 만들어 series에 대입
series =series[5<=series][series<10]
print(series)
# orange    5
# banana    8
# dtype: int64
