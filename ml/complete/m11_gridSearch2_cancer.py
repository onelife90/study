# RandomForest 적용
# breast_cancer 적용

# 모형 하이퍼 파라미터 튜닝 도구
# GridSearchCV : 그리드를 사용한 복수 하이퍼 파라미터 최적화
# 격자. 그물망을 이용하여 내가 넣은 파라미터에 모두 다 걸리게 하겠다

import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#1-1. 데이터
breast_c = load_breast_cancer()
x = breast_c.data
y = breast_c.target
# print(x.shape)  # (569, 30)
# print(y.shape)  # (569, )

# 1-2. 데이터 전처리
y = np_utils.to_categorical(y)
# print(y.shape)  # (569, 2)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=88, test_size=0.2)

# RandomForestClassifier의 파라미터
parameters = [
    {"n_estimators": [5, 10, 100, 1000], "max_depth": [7,70,700,7000]},
    {"min_samples_leaf": [8,80,800,8000], "min_samples_split":[5,50,500,5000]},
]

kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_jobs=-1)
# 역시 train_test_split의 친구라 model_selection에서 import
# 진짜 모델, 파라미터, cv
# test_size=0.2이기 때문에 test가 별개로 나뉘고 train을 총 5번 돌아가며 훈련을 하겠다

# 훈련
model.fit(x, y)

# 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=7000, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)

# n_estimators=생성할 트리의 개수
# RandomForest는 기본적으로 bootstrap sampling(복원추출)을 사용하며 decision tree 생성으로 algorithm으로 진행

y_pred = model.predict(x_test)

print("최종 정답률 = ", accuracy_score(y_test, y_pred))
# 최종 정답률 =  1.0
