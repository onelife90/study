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

# 문제
import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열 추가
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
print(df)

#range(시작행수, 종료행수-1)
df.index = range(1,11)

# loc[]을 사용하여 df의 2~5행(4개행)과 "banana", "kiwifruit"의 2열을 포함한 DataFrame을 df에 대입
# 첫번째 행의 인덱스는 1, 이후의 인덱스는 정수의 오름차순
df = df.loc[range(2,6), ["banana", "kiwifruit"]]
# loc이므로 이름 참조. range(i,j)=i에서부터 j-1까지. 콤마로 명시! / 컬럼이 2개이므로 리스트 형식으로 묶어줌
print(df)
