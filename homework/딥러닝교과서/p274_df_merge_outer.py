# 결합은 key로 불리는 열을 지정하고 두 데이터베이스의 key값이 일치하는 행을 옆으로 연결
#1) 내부결합 : key열이 공통되지 않은 행을 삭제. 동일한 컬럼이지만 값이 일치하지 않는 행의 경우 남기거나 없앨 수 있음
#2) 외부결합 : key열이 공통되지 않아도 행이 삭제X. nan생성
# 외부결합의 기본
# pd.merge(df1, df2, on=key가 될 컬럼, how="outer")로 생성. 이 경우 df1이 왼쪽에 위치
# key가 아니면서 이름이 같은 열은 접미사가 붙음. 왼쪽 df의 컬럼에는_x가, 오른쪽 df의 컬럼에는 _y가 접미사로 붙음
import pandas as pd
data1 = {"fruits": ["apple","orange", "banana","strawberry","kiwi"],
        "year":[2001,2002,2001,2008,2006],
        "amount": [1,4,5,6,3]}
df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
        "year": [2001,2002,2001,2008,2007],
        "price": [150,120,100,250,3000]}
df2 = pd.DataFrame(data2)
# df1과 df2의 내용 확인
print(df1)
#        fruits  year  amount
# 0       apple  2001       1
# 1      orange  2002       4
# 2      banana  2001       5
# 3  strawberry  2008       6
# 4        kiwi  2006       3
print()
print(df2)
#        fruits  year  price
# 0       apple  2001   150
# 1      orange  2002   120
# 2      banana  2001   100
# 3  strawberry  2008   250
# 4       mango  2007  3000
print()
# df1과 df2의 컬럼 fruits를 key로 하여 외부결합한 df를 df3에 대입
df3 = pd.merge(df1, df2, on="fruits", how="outer")
print(df3)
#        fruits  year_x  amount  year_y   price
# 0       apple  2001.0     1.0  2001.0   150.0
# 1      orange  2002.0     4.0  2002.0   120.0
# 2      banana  2001.0     5.0  2001.0   100.0
# 3  strawberry  2008.0     6.0  2008.0   250.0
# 4        kiwi  2006.0     3.0     NaN     NaN
# 5       mango     NaN     NaN  2007.0  3000.0
