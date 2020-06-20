# 다중분류 아이리스
# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피처수를 줄인다
# 3. regularization

from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 회귀 모델
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
print(x.shape)  # (569, 30)
print(y.shape)  # (569, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=86)

# XGB부스터에 무조건 넣어야 하는 하이퍼파라미터
n_estimators = 250          # 나무의 숫자
learning_rate = 0.09
colsample_bytree = 0.9
colsample_bylevel = 0.5

max_depth = 300
n_jobs = -1

# XGB : 전처리, 결측치 제거 X

#2. 모델구성
model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
score = model.score(x_test, y_test)
print("점수 : ", score)
# 점수 :  0.9649122807017544

print(model.feature_importances_)

plot_importance(model)
# plt.show()
