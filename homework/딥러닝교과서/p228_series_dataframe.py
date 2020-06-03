import pandas as pd

# series용 라벨(인덱스) 작성
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# series용 데이터 대입
data = [10,5,8,12,3]

# series 작성
series = pd.Series(data, index=index)

# 딕셔너리형을 사용하여 Dataframe용 데이터 작성
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001,2002,2001,2008,2006],
        "time": [1,4,5,6,3]}

# Dataframe 작성
df = pd.DataFrame(data)

print("Series 데이터")
print(series)
# Series 데이터
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64
print("DataFrame 데이터")
print(df)
# DataFrame 데이터
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3
