# XGB도 역시 evaulate가 있다

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=99)
# print(x_train.shape)    # (30, 4)
# print(x_test.shape)     # (120, 4)
# print(y_train.shape)    # (30, )
# print(y_test.shape)     # (120, )

#2. 모델 구성
model = XGBClassifier(n_estimators=1000, learning_rate=0.1)
# n_estimators는 딥러닝의 epochs와 같음

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric="merror", eval_set=[(x_train,y_train), (x_test, y_test)],
         early_stopping_rounds=20)
# verbose 딥러닝의 metrics가 있었음. 머신러닝의 지표는 rmse, mae, logloss, error(=acc), auc(정확도 acc의 친구)
# error가 0.8이면 acc가 0.2

#4. 평가
result = model.evals_result()
print("evals_result : \n", result)
# evals_result :
#  {'validation_0': {'merror': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, 
# 'validation_1': {'merror': [0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333, 0.058333]}}
# validation_0 == (x_train,y_train)의 결과
# validation_1 == (x_test, y_test)의 결과

#5. 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("acc : ", acc)
# acc :  0.9416666666666667
