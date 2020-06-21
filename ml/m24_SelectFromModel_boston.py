from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#1. 데이터
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=99)

#2. 원모델 구성
model = XGBRegressor()

#3. 원모델 훈련
model.fit(x_train,y_train)

#4-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
# print(thresholds)

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.001,0.01], "colsample_bytree":[0.6,0.9,1], "max_depth":[4,5,6]},
    {"n_estimators":[500,700,900], "learning_rate":[0.3,0.5,0.7], "colsample_bytree":[0.6,0.9,1],
     "max_depth":[4,5,6], "colsample_bylevel":[0.6,0.7,0.9]}
]

#4-2. SelectFromModel 생성
# 컬럼수 만큼 돈다!
# 중요하지 않은 컬럼을 삭제
# 그 기준은 위에 thresholds = np.sort(model.feature_importances_)에서 중요도가 낮은 순으로 정렬을 해놓았음
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 또다른 파라미터 median
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    # select에 들어갈 모델 구성
    selection_model = GridSearchCV(XGBRegressor(), parameters, cv=3, n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

# (101, 13)
# Thresh=0.001, n=13, R2: 82.38%
# (101, 12)
# Thresh=0.003, n=12, R2: 78.85%
# (101, 11)
# Thresh=0.005, n=11, R2: 81.00%
# (101, 10)
# Thresh=0.007, n=10, R2: 83.20%
# (101, 9)
# Thresh=0.011, n=9, R2: 83.11%
# (101, 8)
# Thresh=0.013, n=8, R2: 81.87%
# (101, 7)
# Thresh=0.034, n=7, R2: 79.85%
# (101, 6)
# Thresh=0.041, n=6, R2: 80.06%
# (101, 5)
# Thresh=0.043, n=5, R2: 79.44%
# (101, 4)
# Thresh=0.046, n=4, R2: 76.20%
# (101, 3)
# Thresh=0.079, n=3, R2: 72.92%
# (101, 2)
# Thresh=0.293, n=2, R2: 67.96%
# (101, 1)
# Thresh=0.422, n=1, R2: 46.88%

# 과제: 그리드 서치까지 엮어라
# 데이콘 71개 컬럼 적용해서 제출
# 메일 제목 : 말똥이 24등
