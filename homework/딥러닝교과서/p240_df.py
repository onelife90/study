# Dataframe : Series를 여러개 묶은 것 같은 2차원 데이터 구조
# pd.Dataframe()에 Series를 전달하여 생성
# 행에는 0부터 오름차순으로 번호가 붙음
# Dataframe의 값으로 딕셔너리형(리스트포함)을 넣어도 됨. 해당 리스트형의 길이는 동일해야함
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
# series1과 series2로 Dataframe을 생성하여 df에 대입
df = pd.DataFrame([series1, series2])
print(df)
#    apple  orange  banana  strawberry  kiwi
# 0     10       5       8          12     3
# 1     30      25      12          10     8
# DataFrame에 series를 전달 시 리스형으로 묶어줘야함 ㅠㅠ
