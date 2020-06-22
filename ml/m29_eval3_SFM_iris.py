'''
1. 회귀
2. 이진 분류
3. 다중 분류

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlyStopping 적용
3. plt로 그릴 것
4. SelectFromModel에 적용
4. 결과는 주석으로 소스 하단에 표시
5. m27~29까지 완벽 이해할 것!
'''

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
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = XGBClassifier(n_estimators=3, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=True, eval_metric=["merror", "mlogloss"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=3)
    
    y_pred = selection_model.predict(selection_x_test)
    
    results = selection_model.evals_result()
    print("evals_result : \n", results)
    
    score = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
