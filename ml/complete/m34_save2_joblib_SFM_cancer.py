from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import dump, load
import joblib

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=88)

#2. 모델 구성
model = XGBClassifier(n_estimators=500, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric=["error", "logloss"], 
        eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=30)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  0.00000000e+00 0.00000000e+00 1.59294781e-04 2.31796945e-03
#  3.45580792e-03 6.28038170e-03 7.32157705e-03 8.93414300e-03
#  1.04927327e-02 1.08814165e-02 1.45258689e-02 1.51193989e-02
#  2.47907490e-02 2.69612111e-02 4.51508127e-02 4.56456952e-02
#  2.57064790e-01 5.20898163e-01]

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = XGBClassifier(n_estimators=500, max_depth=4, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=False, eval_metric=["error", "logloss"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=50)
    
    y_pred = selection_model.predict(selection_x_test)
    
    # results = selection_model.evals_result()
    # print("evals_result : \n", results)
    
    score = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
# (455, 17)
# Thresh=0.008, n=17, acc: 99.12%

# 모델 저장
joblib.dump(selection_model, "./model/xgb_save/cancer_acc_99.12_joblib.dat")
