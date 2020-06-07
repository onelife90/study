# df에 새로운 관측 데이터나 거래 정보를 얻었을 때 기존에 추가
# df.append("Series형 데이터", ignore_index=True)
# 인덱스가 일치하지 않으면 df에 새로운 컬럼이 추가되고 값이 존재하지 않는 요소는 nan이 됨
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
data3 = [30,12,10,8,25,3]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
df = pd.DataFrame([series1, series2])
# df에 seires3를 추가하고 df에 다시 대입
index.append("pineapple")
series3 = pd.Series(data3, index=index)
df = pd.DataFrame([series1, series2])
# df에 다시 대입
df = df.append(series3, ignore_index=True)
print(df)
#    apple  orange  banana  strawberry  kiwi  pineapple
# 0     10       5       8          12     3        NaN
# 1     30      25      12          10     8        NaN
# 2     30      12      10           8    25        3.0

# 열 추가 df["새로운 컬럼"]으로 Series 또는 리스트를 대입해서 새 열 추가
# 리스트를 대입하면 첫 행부터 순소대로 요소가 할당, Series를 대입하면 Series의 인덱스가 df의 인덱스에 대응
import  pandas as pd
index = ["apple","orange","banana","strawberry","kiwi"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
new_col = pd.Series([15,7], index=[0,1])
df = pd.DataFrame([series1, series2])
# df에 새로운 열 "mango"를 만들어 new_col의 데이터 추가
df["mango"] = new_col
print(df)
#    apple  orange  banana  strawberry  kiwi  mango
# 0     10       5       8          12     3     15
# 1     30      25      12          10     8      7
