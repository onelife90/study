# 주석 부분을 코드로 작성
import pandas as pd
import numpy as np
from numpy import nan as NA

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df.columns = ["","Alchol","Malic acid", "Ash", "Alcalinity of ash","Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
              "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines", "Proline"]

# 변수 df의 상위 10행을 변수 df_ten에 대입
df_ten = df.head(10)
# print(df_ten)
#       Alchol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  0D280/0D315 of diluted wines  Proline
# 0  1   14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1  1   13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2  1   13.16        2.36  2.67               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3  1   14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4  1   13.24        2.59  2.87               21.0        118           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93      735
# 5  1   14.20        1.76  2.45               15.2        112           3.27        3.39                  0.34             1.97             6.75  1.05                          2.85     1450
# 6  1   14.39        1.87  2.45               14.6         96           2.50        2.52                  0.30             1.98             5.25  1.02                          3.58     1290
# 7  1   14.06        2.15  2.61               17.6        121           2.60        2.51                  0.31             1.25             5.05  1.06                          3.58     1295
# 8  1   14.83        1.64  2.17               14.0         97           2.80        2.98                  0.29             1.98             5.20  1.08                          2.85     1045
# 9  1   13.86        1.35  2.27               16.0         98           2.98        3.15                  0.22             1.85             7.22  1.01                          3.55     1045

# 데이터 일부 누락
df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
# print(df_ten)
#         Alchol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  0D280/0D315 of diluted wines  Proline
# 0  1.0   14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1  NaN   13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2  1.0   13.16        2.36   NaN               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3  1.0   14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4  1.0   13.24        2.59  2.87               21.0        118           2.80        2.69                   NaN             1.82             4.32  1.04                          2.93      735
# 5  1.0   14.20        1.76  2.45               15.2        112           3.27        3.39                  0.34             1.97             6.75  1.05                          2.85     1450
# 6  1.0   14.39        1.87  2.45               14.6         96           2.50        2.52                  0.30             1.98             5.25  1.02                          3.58     1290
# 7  1.0   14.06        2.15   NaN               17.6        121           2.60        2.51                  0.31             1.25             5.05  1.06                          3.58     1295
# 8  1.0   14.83        1.64  2.17               14.0         97           2.80        2.98                  0.29             1.98             5.20  1.08                          2.85     1045
# 9  1.0   13.86        1.35  2.27               16.0         98           2.98        3.15                  0.22             1.85             7.22  1.01                          3.55     1045

# fillna()메서드로 nan 부부분에 열의 평균값 대입
df_ten_mean = df_ten.fillna(df_ten.mean())
# print(df_ten_mean)
#         Alchol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  0D280/0D315 of diluted wines  Proline
# 0  1.0   14.23        1.71  2.43               15.6        127           2.80        3.06              0.280000             2.29             5.64  1.04                          3.92     1065
# 1  1.0   13.20        1.78  2.14               11.2        100           2.65        2.76              0.260000             1.28             4.38  1.05                          3.40     1050
# 2  1.0   13.16        2.36  2.41               18.6        101           2.80        3.24              0.300000             2.81             5.68  1.03                          3.17     1185
# 3  1.0   14.37        1.95  2.50               16.8        113           3.85        3.49              0.240000             2.18             7.80  0.86                          3.45     1480
# 4  1.0   13.24        2.59  2.87               21.0        118           2.80        2.69              0.282222             1.82             4.32  1.04                          2.93      735
# 5  1.0   14.20        1.76  2.45               15.2        112           3.27        3.39              0.340000             1.97             6.75  1.05                          2.85     1450
# 6  1.0   14.39        1.87  2.45               14.6         96           2.50        2.52              0.300000             1.98             5.25  1.02                          3.58     1290
# 7  1.0   14.06        2.15  2.41               17.6        121           2.60        2.51              0.310000             1.25             5.05  1.06                          3.58     1295
# 8  1.0   14.83        1.64  2.17               14.0         97           2.80        2.98              0.290000             1.98             5.20  1.08                          2.85     1045
# 9  1.0   13.86        1.35  2.27               16.0         98           2.98        3.15              0.220000             1.85             7.22  1.01                          3.55     1045

# "Alcohol"열의 평균 출력
al_mean = df_ten["Alchol"].mean()
# print(al_mean)
# 13.954000000000002

# 중복된 행 제거
df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
drop_df_ten = df_ten.drop_duplicates()
# print(drop_df_ten)

# "Alchol"열의 구간 리스트 작성
alcohol_bins = [0,5,10,15,20,25]
alcohol_cut_data = pd.cut(df_ten["Alchol"], alcohol_bins)

# 구간수 집계 출력
print(pd.value_counts(alcohol_cut_data))
# (10, 15]    10
# (20, 25]     0
# (15, 20]     0
# (5, 10]      0
# (0, 5]       0
# Name: Alchol, dtype: int64
