from sklearn.feature_selection import SelectFromModel
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=88)

#2. 모델 구성
model = LGBMClassifier(n_estimators=1000, n_jobs=-1, objective="multiclass")

#3. 훈련
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric=["multi_error", "multi_logloss"], early_stopping_rounds=100)
score = model.score(x_test, y_test)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.01818451 0.01885792 0.3417337  0.62122387]

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = LGBMClassifier(n_estimators=1000, max_depth=4, learning_rate=0.5, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train,verbose=False, eval_metric=["multi_error", "multi_logloss"], eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=100)
    
    y_pred = selection_model.predict(selection_x_test)
    
    # results = selection_model.evals_result()
    # print("evals_result : \n", results)
    
    score = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
# (120, 1)
# Thresh=285.000, n=4, acc: 100.00%

model.save_model("./model/xgb_save/lgbm_iris_acc_100_model")
