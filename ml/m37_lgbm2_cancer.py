from sklearn.feature_selection import SelectFromModel
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=88)

#2. 모델 구성
model = LGBMClassifier(n_estimators=500, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric=["error", "logloss"], 
        eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=30)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = LGBMClassifier(n_estimators=500, max_depth=4, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=False, eval_metric=["error", "logloss"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=50)
    
    y_pred = selection_model.predict(selection_x_test)
    
    # results = selection_model.evals_result()
    # print("evals_result : \n", results)
    
    score = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
# (455, 8)
# Thresh=77.000, n=8, acc: 100.00%

# 모델 저장
pickle.dump(selection_model, open("./model/xgb_save/lgbm_cancer_acc_100_pickle.dat", "wb"))
