from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=88)

#2. 모델 구성
model = XGBClassifier(n_estimators=1000, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train, eval_metric="merror", eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=100)
score = model.score(x_test, y_test)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.07822347 0.18346179 0.25513232 0.48318246]

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = XGBClassifier(n_estimators=1000, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=True, eval_metric=["merror", "mlogloss"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=80)
    
    y_pred = selection_model.predict(selection_x_test)
    
    results = selection_model.evals_result()
    print("evals_result : \n", results)
    
    score = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
# Thresh=0.483, n=1, acc: 96.67%

model.save_model("./model/xgb_save/iris_acc_96.67_model")
