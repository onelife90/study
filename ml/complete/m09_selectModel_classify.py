import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

# warnigs라는 에러를 그냥 넘어가겠다
warnings.filterwarnings('ignore')

#1-1. csv 파일 불러오기
iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
# print(x)    # [150 rows x 4 columns]
# print(y)    # Name: virginica, Length: 150, dtype: int64

#1-2. train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, test_size=0.2)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# all_estimator에 사이킷런의 모든 분류 모델이 저장
# ※ 주의! LogisticRegression는 분류 모델임※

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 = ", accuracy_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)
# AdaBoostClassifier 의 정답률 =  0.9333333333333333
# BaggingClassifier 의 정답률 =  0.9333333333333333
# BernoulliNB 의 정답률 =  0.26666666666666666
# CalibratedClassifierCV 의 정답률 =  0.9333333333333333
# ComplementNB 의 정답률 =  0.6
# DecisionTreeClassifier 의 정답률 =  0.9333333333333333
# ExtraTreeClassifier 의 정답률 =  0.9333333333333333
# ExtraTreesClassifier 의 정답률 =  0.9333333333333333
# GaussianNB 의 정답률 =  0.9333333333333333
# GaussianProcessClassifier 의 정답률 =  0.9333333333333333
# GradientBoostingClassifier 의 정답률 =  0.9333333333333333
# KNeighborsClassifier 의 정답률 =  0.9333333333333333
# LabelPropagation 의 정답률 =  0.9333333333333333
# LabelSpreading 의 정답률 =  0.9333333333333333
# LinearDiscriminantAnalysis 의 정답률 =  0.9666666666666667
# LinearSVC 의 정답률 =  0.9333333333333333
# LogisticRegression 의 정답률 =  0.9333333333333333
# LogisticRegressionCV 의 정답률 =  0.9333333333333333
# MLPClassifier 의 정답률 =  0.9333333333333333
# MultinomialNB 의 정답률 =  0.9333333333333333
# NearestCentroid 의 정답률 =  0.9666666666666667
# NuSVC 의 정답률 =  0.9333333333333333
# PassiveAggressiveClassifier 의 정답률 =  0.6666666666666666
# Perceptron 의 정답률 =  0.6
# QuadraticDiscriminantAnalysis 의 정답률 =  0.9666666666666667
# RadiusNeighborsClassifier 의 정답률 =  0.9666666666666667
# RandomForestClassifier 의 정답률 =  0.9333333333333333
# RidgeClassifier 의 정답률 =  0.8333333333333334
# RidgeClassifierCV 의 정답률 =  0.8333333333333334
# SGDClassifier 의 정답률 =  0.8333333333333334
# SVC 의 정답률 =  0.9666666666666667
# 0.20.1
