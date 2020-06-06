# 모형 하이퍼 파라미터 튜닝 도구
# GridSearchCV : 그리드를 사용한 복수 하이퍼 파라미터 최적화
# 격자. 그물망을 이용하여 내가 넣은 파라미터에 모두 다 걸리게 하겠다

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#1-1. 데이터 csv 파일 불러오기
iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
# print(x)    # [150 rows x 4 columns]
# print(y)    # Name: virginica, Length: 150, dtype: int64

# 1-2. train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, test_size=0.2)

# SVC의 파라미터
parameters = [
    {"C": [1,10,100,1000], "kernel": ["linear"]},
    {"C": [1,10,100,1000], "kernel": ["rbf"], "gamma":[0.001, 0.0001]},
    {"C": [1,10,100,1000], "kernel": ["sigmoid"], "gamma":[0.001, 0.0001]},
]
# kernel : 수학적 기교를 사용하여 스칼라 곱 계산. 고차원 공간을 매핑하는데 많이 사용
# rbf=가우시안 커널. 차원이 무한한 특성 공간에 매핑
# gamma=커널의 폭을 제어하는 매개변수

kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(SVC(), parameters, cv=kfold)
# 역시 train_test_split의 친구라 model_selection에서 import
# 진짜 모델, 파라미터, cv
# test_size=0.2이기 때문에 test가 별개로 나뉘고 train을 총 5번 돌아가며 훈련을 하겠다

# 훈련
model.fit(x_train, y_train)

# 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)

y_pred = model.predict(x_test)

print("최종 정답률 = ", accuracy_score(y_test, y_pred))
# 최종 정답률 =  0.9666666666666667
