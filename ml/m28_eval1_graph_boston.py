# 그래프 그리기
# 머신러닝의 지표는 rmse, mae, logloss, error(=acc), auc(정확도 acc의 친구)
# 딥러닝의 earlyStopping을 머신러닝에 적용해보자
# fit에 callbacks에 있었음
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
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#1. 데이터
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=88)

#2. 모델 구성
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
# n_estimators는 딥러닝의 epochs와 같음

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric=["rmse", "logloss"], eval_set=[(x_train,y_train), (x_test, y_test)],
         early_stopping_rounds=100)

# verbose 딥러닝의 metrics가 있었음. 머신러닝의 지표는 rmse, mae, logloss, error(=acc), auc(정확도 acc의 친구)
# error가 0.8이면 acc가 0.2
# early_stopping_rounds 실행시 Stopping. Best iteration: 가 verbose에 출력
# validation_1 == (x_test, y_test)의 결과이기 때문에 test를 기준으로 평가가 됨

#4. 평가
result = model.evals_result()
print("evals_result : \n", result)
# evals_result :
#  {'validation_0': {'rmse': [22.09964, 20.094713, 18.289314], 'logloss': [-810.254639, -810.254639, -810.254639]}, 
#  'validation_1': {'rmse': [21.539825, 19.548641, 17.804596], 'logloss': [-789.069214, -789.069214, -789.069214]}}
# 두가지 이상의 지표를 쓰려면 []형태로 묶어줌
# validation_0 == (x_train,y_train)의 결과
# validation_1 == (x_test, y_test)의 결과

# 5. 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("R2 : ", r2)
# R2 :  0.823625251495531

#6. 시각화
epochs = len(result['validation_0']['mae'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mae'], label='Train')
ax.plot(x_axis, result['validation_1']['mae'], label='Test')
ax.legend()
plt.ylabel('MAE')
plt.title('XGBoost MAE')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label='Train')
ax.plot(x_axis, result['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()
