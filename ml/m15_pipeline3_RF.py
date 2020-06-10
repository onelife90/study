# pipeline : 전처리도 한 방에 처리하는 자동화
# RandomizedSearchCV + Pipeline
# RF

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=88)

# 그리드/랜덤 서치에서 사용할 매개 변수
# parameters = [
#     {"svm__C":[1,10,100,1000], "svm__kernel":['linear']},
#     {"svm__C":[1,10,100], "svm__kernel":['rbf'], "svm__gamma":[0.001,0.0001]},
#     {"svm__C":[1,100,1000], "svm__kernel":['sigmoid'], "svm__gamma":[0.001,0.0001]}
# ]

parameters = [
    {"randomforestclassifier__max_depth":[1,10,100,1000], "randomforestclassifier__n_estimators":[10,100,1000]},
    {"randomforestclassifier__max_depth":[5,50,500,5000], "randomforestclassifier__n_estimators":[7,700,7000], "randomforestclassifier__criterion":["gini"]},
    {"randomforestclassifier__max_depth":[9,90,900,9000], "randomforestclassifier__n_estimators":[90,900,9000], "randomforestclassifier__criterion":["gini", "entropy"]}
]

#2. 모델 구성
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
# SVC 모델과 MinMaxScaler를 쓰겠다
pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())
model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
acc = model.score(x_test, y_test)
print("최적의 매개 변수 = ", model.best_estimator_)
        #  steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
        #         ('randomforestclassifier',
        #          RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
        #                                 class_weight=None, criterion='gini',
        #                                 max_depth=100, max_features='auto',
        #                                 max_leaf_nodes=None, max_samples=None,
        #                                 min_impurity_decrease=0.0,
        #                                 min_impurity_split=None,
        #                                 min_samples_leaf=1, min_samples_split=2,
        #                                 min_weight_fraction_leaf=0.0,
        #                                 n_estimators=1000, n_jobs=None,
        #                                 oob_score=False, random_state=None,
        #                                 verbose=0, warm_start=False))],
        #  verbose=False)
print("acc: ", acc)
# acc:  0.9666666666666667
