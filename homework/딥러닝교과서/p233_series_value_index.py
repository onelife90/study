# Seies의 데이터와 인덱스 추출
# Series 자료형은 Series.values로 데이터값 참조
# Series 인덱스는 Series.index로 인덱스 참조
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data = [10,5,8,12,3]
series = pd.Series(data, index=index)
# series_values에 series의 데이터 대입
series_values = series.values
# series_index에 seires의 인덱스 대입
series_index = series.index
print(series_values)    # [10  5  8 12  3]
print(series_index)     # Index(['apple', 'orange', 'banana', 'strawberry', 'kiwi'], dtype='object')
