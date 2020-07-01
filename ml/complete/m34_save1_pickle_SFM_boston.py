from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

#1. 데이터
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=99)

#2. 모델 구성
model = XGBRegressor() 
# n_estimators=200, max_depth=4, leraning_rate=0.1, n_jobs=-1

model.fit(x_train, y_train, verbose=True, eval_metric=["rmse", "mae"], eval_set=[(x_train, y_train), (x_test, y_test)])

score = model.score(x_test, y_test)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00282955 0.01017448 0.01068934 0.01081465 0.01164605 0.01270833   
#  0.01729782 0.02083709 0.04275301 0.04288257 0.07989766 0.19983138   
#  0.53763807]

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.6, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=False, eval_metric=["rmse", "mae"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=100)
    
    y_pred = selection_model.predict(selection_x_test)
    
    # results = selection_model.evals_result()
    # print("evals_result : \n", results)
    
    score = r2_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
# (404, 11)
# Thresh=0.011, n=11, R2: 81.14%

# 모델 저장
pickle.dump(selection_model, open("./model/xgb_save/boston_r2_81.14_pickle.dat", "wb"))
