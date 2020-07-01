from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=99)

#2. 모델 구성
model = XGBClassifier(n_estimators=1000, learning_rate=0.1)
# n_estimators는 딥러닝의 epochs와 같음

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric="error", eval_set=[(x_train,y_train), (x_test, y_test)],
         early_stopping_rounds=20)
# verbose 딥러닝의 metrics가 있었음. 머신러닝의 지표는 rmse, mae, logloss, error(=acc), auc(정확도 acc의 친구)
# error가 0.8이면 acc가 0.2

#4. 평가
result = model.evals_result()
print("evals_result : \n", result)
# validation_0 == (x_train,y_train)의 결과
# validation_1 == (x_test, y_test)의 결과

#5. 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("acc : ", acc)

# 모델 저장하기 위해서는 피클이 필요
# 파이썬에서 제공. 데이터를 자료형의 변경없이 파일로 저장
import pickle
pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb"))
# wb == 바이트 형식으로 쓰겠다
print("저장됐다.")

# 모델 불러오기
model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", "rb"))
print("불러왔다.")

y_pred = model2.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("acc : ", acc)
