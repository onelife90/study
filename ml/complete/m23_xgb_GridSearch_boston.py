# 최적의 w를 구하기 위해 optimizer를 사용
# underfitting > 최적의 w > overfitting
# 경사하강법
# 미분을 해서 loss==0이 되는 지점을 찾는 것이 목표
# learning_rate를 작게 주면 자르는 구간이 촘촘해지나 속도가 느림. 줄여서 lr로 표현

# 회귀 모델 보스턴

# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피처수를 줄인다
# 3. regularization

from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# 회귀 모델
x,y=load_boston(return_X_y=True)
# print(x.shape)  #(506, 13)
# print(y.shape)  #(506, )

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
    {"n_estimators":[500,700,900], "learning_rate":[0.3,0.5,0.7], "colsample_bytree":[0.6,0.9,1],
     "max_depth":[4,5,6], "colsample_bylevel":[0.6,0.7,0.9]}
]
# n_jobs = -1

#2. 모델구성
model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
print("best_estimator_ : \n", model.best_estimator_)
print("best_params_ : \n", model.best_params_)

score = model.score(x_test, y_test)
print("점수 : ", score)
# 점수 :  0.9

# best_estimator_ :
#  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='', 
#              learning_rate=0.3, max_delta_step=0, max_depth=4,   
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=500, n_jobs=0, num_parallel_tree=1,    
#              objective='reg:squarederror', random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# best_params_ :
#  {'colsample_bylevel': 0.9, 'colsample_bytree': 0.9, 'learning_rate': 0.3, 'max_depth': 4, 'n_estimators': 500}
# 점수 :  0.8976797534603859
