from sklearn.feature_selection import SelectFromModel
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from joblib import dump, load
import joblib

#1. 데이터
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=67)

#2. 모델 구성
model = LGBMRegressor(n_jobs=-1) 
# n_estimators=200, max_depth=4, leraning_rate=0.1, n_jobs=-1

model.fit(x_train, y_train, verbose=True, eval_metric=["rmse", "mae"], eval_set=[(x_train, y_train), (x_test, y_test)])

score = model.score(x_test, y_test)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [  5   7  47  58  65  75 107 143 150 153 202 222 250]

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = LGBMRegressor(n_estimators=2000, max_depth=3, learning_rate=0.1, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=False, eval_metric=["rmse", "mae"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=1000)
    
    y_pred = selection_model.predict(selection_x_test)
    
    # results = selection_model.evals_result()
    # print("evals_result : \n", results)
    
    score = r2_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
# (404, 13)
# Thresh=10.000, n=13, R2: 87.60%

# 모델 저장
joblib.dump(selection_model, "./model/xgb_save/lgbm_boston_r2_87.60_joblib.dat")
