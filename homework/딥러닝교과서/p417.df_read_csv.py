# CSV(Comma-separated values)라는 데이터 형식 취급. 쉼표로 구분하여 저장한 데이터. 사용이 편리하므로 데이터 분석에 자주 사용
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(df)
#      sepal length  sepal width  petal length  petal width           class
# 0             5.1          3.5           1.4          0.2     Iris-setosa
# 1             4.9          3.0           1.4          0.2     Iris-setosa
# 2             4.7          3.2           1.3          0.2     Iris-setosa
# 3             4.6          3.1           1.5          0.2     Iris-setosa
# 4             5.0          3.6           1.4          0.2     Iris-setosa
# ..            ...          ...           ...          ...             ...
# 145           6.7          3.0           5.2          2.3  Iris-virginica
# 146           6.3          2.5           5.0          1.9  Iris-virginica
# 147           6.5          3.0           5.2          2.0  Iris-virginica
# 148           6.2          3.4           5.4          2.3  Iris-virginica
# 149           5.9          3.0           5.1          1.8  Iris-virginica

# [150 rows x 5 columns]

data = {"OS":["Machintosh", "Windows", "linux"],
        "release":[1984,1985,1991],
        "country":["US", "US", ""]}
df = pd.DataFrame(data)
df.to_csv("OSlist.csv")
