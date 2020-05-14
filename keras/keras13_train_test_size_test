#1. 데이터
import numpy as np
x = np.array(range(1,101)) # 1부터 시작해서 101-1까지 나열
y = np.array(range(101,201)) # y=wx+b / w=1, b=100이 되는 함수

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.5, test_size=0.4
)       

print(x_train)
# print(x_val)
print(x_test)

# train_size=0.7, test_size=0.4
# 에러가 뜸. Reduce test_size and/or train_size.
# train_size + test_size < 1

# 그렇다면 train_size + test_size > 1 경우는?
# train_size=0.5, test_size=0.4
#[97 77 63 19 14 60 64 47 52 84 35 12 66 58 16 70 87 41 54 72 20 27 13 75
 81 38 28 51 61 50 82  9 55 67 36  8 21 85 46 62 11 56 89 32 93 10  5 73
 39 95] => 50개
#[37  7 59 91 83 29 99 30 26 76 98 65 96 68  4 44 88 40 69 17 34 53 71 80
 24 92 25 49 78  1 57 31 23 79 43 18 48 42  3 33] => 40개
 
# train_size:test_size = 5:4의 비율로 데이터 반환. 손실된 데이터(6,15,22,45,62,....) 존재
