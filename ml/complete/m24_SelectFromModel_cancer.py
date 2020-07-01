from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=99)

#2. 모델 구성
model = XGBClassifier()
model.fit(x_train,y_train)

# 컬럼수 만큼 돌 thresholds 생성
# thresholds는 '문턱' ex) 반올림
thresholds = np.sort(model.feature_importances_)
print(thresholds)

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.001,0.01], "colsample_bytree":[0.6,0.9,1], "max_depth":[4,5,6]},
    {"n_estimators":[500,700,900], "learning_rate":[0.3,0.5,0.7], "colsample_bytree":[0.6,0.9,1],
     "max_depth":[4,5,6], "colsample_bylevel":[0.6,0.7,0.9]}
]

# 컬럼수 만큼 돈다!
# 중요하지 않은 컬럼은 제쳐두고 중요도를 비교하여 문턱을 넘는다
# 그 기준은 위에 thresholds = np.sort(model.feature_importances_)에서 중요도가 낮은 순으로 정렬을 해놓았음
for thresh in thresholds:
    selection = SelectFromModel(model,threshold=thresh, prefit=True) # 또다른 파라미터 median
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    # select에 들어갈 모델 구성
    selection_model = GridSearchCV(XGBClassifier(), parameters, cv=3, n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, acc : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 30)
# Thresh=0.000, n=30, acc : 94.52%
# (113, 17)
# Thresh=0.003, n=17, acc : 94.96%
# (113, 16)
# Thresh=0.003, n=16, acc : 93.20%
# (113, 15)
# Thresh=0.005, n=15, acc : 94.96%
# (113, 14)
# Thresh=0.006, n=14, acc : 94.08%
# (113, 13)
# Thresh=0.009, n=13, acc : 92.98%
# (113, 12)
# Thresh=0.009, n=12, acc : 95.83%
# (113, 11)
# Thresh=0.009, n=11, acc : 94.96%
# (113, 10)
# Thresh=0.011, n=10, acc : 92.11%
# (113, 9)
# Thresh=0.011, n=9, acc : 94.52%
# (113, 8)
# Thresh=0.013, n=8, acc : 94.08%
# (113, 7)
# Thresh=0.016, n=7, acc : 95.18%
# (113, 6)
# Thresh=0.024, n=6, acc : 94.08%
# (113, 5)
# Thresh=0.042, n=5, acc : 93.20%
# (113, 4)
# Thresh=0.052, n=4, acc : 93.42%
# (113, 3)
# Thresh=0.202, n=3, acc : 94.08%
# (113, 2)
# Thresh=0.286, n=2, acc : 94.08%
# (113, 1)
# Thresh=0.297, n=1, acc : 91.01%
