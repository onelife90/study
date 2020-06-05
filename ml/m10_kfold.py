# kfold 교차검증
# train 학습용 데이터 집합 내 종속 변수 값을 잘 예측하였는지 나타내는 성능. 표본내 성능 검증
# 조각 조각 내어서 train과 test를 분할
# kfold cv=5 / 데이터 셋을 5개 조각을 내고 5번 훈련을 함
# train4 : test1

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

# warnigs라는 에러를 그냥 넘어가겠다
warnings.filterwarnings('ignore')

#1-1. 데이터 csv 파일 불러오기
iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
# print(x)    # [150 rows x 4 columns]
# print(y)    # Name: virginica, Length: 150, dtype: int64

#1-2. train_test_split 
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, test_size=0.2)

#1-2. KFold(train_test_split 친구)
kfold = KFold(n_splits=5, shuffle=True)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# all_estimator에 사이킷런의 모든 분류 모델이 저장
# ※ 주의! LogisticRegression는 분류 모델임※

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    scores = cross_val_score(model, x,y, cv=kfold)
    print(name, "의 정답률 = ")
    print(scores)

import sklearn
print(sklearn.__version__)
# AdaBoostClassifier 의 정답률 = 
# [0.93333333 0.93333333 0.96666667 0.9        0.9       ]
# BaggingClassifier 의 정답률 = 
# [0.93333333 1.         0.96666667 0.93333333 0.93333333]
# BernoulliNB 의 정답률 = 
# [0.23333333 0.16666667 0.2        0.3        0.16666667]
# CalibratedClassifierCV 의 정답률 = 
# [0.86666667 0.93333333 0.96666667 0.93333333 0.86666667]
# ComplementNB 의 정답률 = 
# [0.66666667 0.56666667 0.7        0.73333333 0.66666667]
# DecisionTreeClassifier 의 정답률 = 
# [0.9        1.         0.93333333 1.         0.96666667]
# ExtraTreeClassifier 의 정답률 = 
# [0.93333333 0.9        0.9        0.9        0.9       ]
# ExtraTreesClassifier 의 정답률 = 
# [0.96666667 0.96666667 1.         0.96666667 0.96666667]
# GaussianNB 의 정답률 = 
# [0.96666667 0.93333333 0.93333333 0.93333333 0.96666667]
# GaussianProcessClassifier 의 정답률 = 
# [0.96666667 0.96666667 0.93333333 1.         0.9       ]
# GradientBoostingClassifier 의 정답률 = 
# [0.9        0.96666667 0.96666667 0.96666667 0.96666667]
# KNeighborsClassifier 의 정답률 = 
# [0.96666667 0.93333333 0.93333333 0.96666667 1.        ]
# LabelPropagation 의 정답률 = 
# [0.96666667 0.83333333 0.96666667 0.96666667 0.96666667]
# LabelSpreading 의 정답률 = 
# [1.         0.96666667 0.93333333 0.96666667 0.93333333]
# LinearDiscriminantAnalysis 의 정답률 = 
# [0.96666667 0.96666667 0.93333333 1.         1.        ]
# LinearSVC 의 정답률 = 
# [1.         0.96666667 0.93333333 0.83333333 1.        ]
# LogisticRegression 의 정답률 = 
# [0.86666667 0.96666667 1.         0.93333333 1.        ]
# LogisticRegressionCV 의 정답률 = 
# [0.93333333 0.83333333 0.96666667 0.83333333 1.        ]
# MLPClassifier 의 정답률 = 
# [0.96666667 0.96666667 0.96666667 0.96666667 0.96666667]
# MultinomialNB 의 정답률 = 
# [0.86666667 0.96666667 0.63333333 0.8        0.9       ]
# NearestCentroid 의 정답률 = 
# [0.96666667 0.93333333 0.83333333 0.96666667 0.9       ]
# NuSVC 의 정답률 =
# [1.         0.93333333 0.93333333 0.96666667 0.96666667]
# PassiveAggressiveClassifier 의 정답률 =
# [0.7        0.8        0.83333333 0.6        0.73333333]
# Perceptron 의 정답률 =
# [0.73333333 0.96666667 0.56666667 0.56666667 0.5       ]
# QuadraticDiscriminantAnalysis 의 정답률 =
# [0.93333333 1.         1.         0.93333333 0.96666667]
# RadiusNeighborsClassifier 의 정답률 =
# [0.93333333 1.         1.         0.9        0.96666667]
# RandomForestClassifier 의 정답률 =
# [0.9        0.96666667 0.93333333 1.         0.96666667]
# RidgeClassifier 의 정답률 =
# [0.8        0.96666667 0.73333333 0.83333333 0.83333333]
# RidgeClassifierCV 의 정답률 =
# [0.93333333 0.8        0.73333333 0.9        0.83333333]
# SGDClassifier 의 정답률 =
# [0.56666667 0.83333333 0.33333333 0.93333333 0.86666667]
# SVC 의 정답률 =
# [0.96666667 1.         0.96666667 0.96666667 1.        ]
# 0.20.1
