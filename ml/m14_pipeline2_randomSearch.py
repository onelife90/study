# pipeline : 전처리도 한 방에 처리하는 자동화
# RandomizedSearchCV + Pipeline

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=88)

# 그리드/랜덤 서치에서 사용할 매개 변수
parameters = [
    {"svm__C":[1,10,100,1000], "svm__kernel":['linear']},
    {"svm__C":[1,10,100], "svm__kernel":['rbf'], "svm__gamma":[0.001,0.0001]},
    {"svm__C":[1,100,1000], "svm__kernel":['sigmoid'], "svm__gamma":[0.001,0.0001]}
]

#2. 모델 구성
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
# SVC 모델과 MinMaxScaler를 쓰겠다
model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
acc = model.score(x_test, y_test)
print("최적의 매개 변수 = ", model.best_estimator_)
# 최적의 매개 변수 =  Pipeline(memory=None,
#          steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
#                 ('svm',
#                  SVC(C=10, break_ties=False, cache_size=200, class_weight=None,
#                      coef0=0.0, decision_function_shape='ovr', degree=3,
#                      gamma='scale', kernel='linear', max_iter=-1,
#                      probability=False, random_state=None, shrinking=True,
#                      tol=0.001, verbose=False))],
#          verbose=False)
print("acc: ", acc)
# acc:  1.0
