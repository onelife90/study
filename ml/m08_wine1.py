# 머신러닝 기법을 적용하여 모델 완성
# wine.csv 마지막 컬럼 y. 슬라이싱
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score

#1-1. 데이터 읽어오기
wine = pd.read_csv('./data/csv/winequality-white.csv', index_col=None, header=0, sep=';', encoding='CP949')
# print(wine)   # [4898 rows x 12 columns]
# wine.info()
 #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         4898 non-null   float64
#  1   volatile acidity      4898 non-null   float64
#  2   citric acid           4898 non-null   float64
#  3   residual sugar        4898 non-null   float64
#  4   chlorides             4898 non-null   float64
#  5   free sulfur dioxide   4898 non-null   float64
#  6   total sulfur dioxide  4898 non-null   float64
#  7   density               4898 non-null   float64
#  8   pH                    4898 non-null   float64
#  9   sulphates             4898 non-null   float64
#  10  alcohol               4898 non-null   float64
#  11  quality               4898 non-null   int64
# dtypes: float64(11), int64(1)

#1-2. 데이터 x,y 자르기
x = wine.iloc[:, 0:11]
y = wine.iloc[:, 11]
# print(x)
# print(x.shape)  # (4898, 11)
# print(y)
# print(y.shape)  # (4898, )

#1-3. 데이터 전처리
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, train_size=0.6)
# print(x_train.shape)    # (2938, 11)
# print(x_test.shape)     # (1960, 11)
# print(y_train.shape)    # (2938,)
# print(y_test.shape)     # (1960,)

#2-1. 모델 구성
# model = RandomForestClassifier()
# acc:  0.6576530612244897
# R2:  0.3904680472546548

model = KNeighborsClassifier(n_neighbors=1)
# acc:  0.6035714285714285
# R2:  0.026911294951998843

# model = LinearSVC()
# acc:  0.5280612244897959
# R2:  0.1465201149474823

# model = SVC()
# acc:  0.5071428571428571
# R2:  0.07826875438508785

#2-2. 훈련
model.fit(x_train, y_train)

#2-3. 평가, 예측
y_pred = model.predict(x_test)
acc  = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("acc: ", acc)
print("R2: ", r2)
