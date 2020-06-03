# DataFrame의 데이터는 행과 열을 지정해서 참조
# loc는 이름으로 참조, iloc는 번호로 참조
# 이름으로 참조? DataFrame형 데이터를 이름(인덱스, 컬럼명)으로 참조한다는 말 ==> loc 사용
# 변수 df를 df.loc["인덱스 리스트", "컬럼 리스트"]로 지정하여 해당 범위의 DataFrame을 얻을 수 있다
import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001,2002,2001,2008,2006],
        "time": [1,4,5,6,3]}
df = pd.DataFrame(data)
print("df: \n", df)
#         fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

# 이름에 의한 참조
df = df.loc[[1,2], ["time", "year"]]
print("df: \n", df)
#     time  year
# 1     4  2002
# 2     5  2001
