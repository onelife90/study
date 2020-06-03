# pandas : 데이터 셋을 다루는 라이브러리. 일반적인 데이터베이스에서 이뤄지는 작업을 수행
# 수치뿐 아니라 이름과 주소 등 문자열 데이터도 쉽게 처리
# 두 가지 데이터 구조가 존재. Series와 Dataframe
# 각 행과 열에는 라벨이 부여. 행 라벨(인덱스), 열 라벨(컬럼)
# series는 라벨이 붙은 1차원 데이터, Dataframe 여러 시리즈를 묶은 것과 같은 2차원 데이터 구조

import pandas as pd

# Series 데이터
fruits = {"orange":2, "bananan":3}
print(pd.Series(fruits))
# orange     2
# bananan    3
# dtype: int64

# Dataframe 데이터
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "Year": [2001,2002,2001,2008,2006],
        "time": [1,4,5,6,3]}
df = pd.DataFrame(data)
print(df)
#        fruits  Year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3
