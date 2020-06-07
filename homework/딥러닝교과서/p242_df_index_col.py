# df에서 행의 이름을 인덱스, 열의 이름을 컬럼
# 인수를 지정하지 않으면 0부터 오름차순으로 인덱스 할당
# 컬럼은 원본 데이터 Series의 인덱스 및 딕셔너리형의 키가 됨
# df.index에 행 수와 같은 길이의 리스트를 대입하여 설정
# df.columns에 열 수와 같은 길이의 리스트 대입하여 설정
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
df = pd.DataFrame([series1, series2])
# df의 인덱스가 1부터 시작하도록 설정
df.index = [1,2]
print(df)
#    apple  orange  banana  strawberry  kiwi
# 1     10       5       8          12     3
# 2     30      25      12          10     8
