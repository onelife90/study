# 최적의 w를 구하기 위해 optimizer를 사용
# underfitting > 최적의 w > overfitting
# 경사하강법
# 미분을 해서 loss==0이 되는 지점을 찾는 것이 목표
# learning_rate를 작게 주면 자르는 구간이 촘촘해지나 속도가 느림. 줄여서 lr로 표현

# 다중분류 아이리스

# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피처수를 줄인다
# 3. regularization

from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# 회귀 모델
iris = load_iris()
x = iris.data
y = iris.target
# print(x.shape)  # (150, 4)
# print(y.shape)  # (150, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=86)

# XGB부스터에 무조건 넣어야 하는 하이퍼파라미터
n_estimators = 100          # 나무의 숫자
learning_rate = 0.59
colsample_bytree = 0.7
colsample_bylevel = 0.5

max_depth = 4
n_jobs = -1

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.001,0.01], "colsample_bytree":[0.6,0.9,1], "max_depth":[4,5,6]},
    {"n_estimators":[500,700,900], "learning_rate":[0.1,0.001,0.01], "colsample_bytree":[0.6,0.9,1],
     "max_depth":[4,5,6], "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

#2. 모델구성
xgb = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel)

model = GridSearchCV(XGBClassifier(), parameters, cv=5, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
# model.best_estimator_
print("best_estimator_ : \n", model.best_estimator_)

# model.best_params_
print("best_params_ : \n", model.best_params_)

score = model.score(x_test, y_test)
print("점수 : ", score)
# 점수 :  0.9
