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
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=88)

#2. 모델 구성
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

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

    selection_model = XGBClassifier(n_estimators=3, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=True, eval_metric=["rmse", "mae"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=3)
    
    y_pred = selection_model.predict(selection_x_test)
    
    results = selection_model.evals_result()
    print("evals_result : \n", results)
    
    score = r2_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))

# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 30)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result :
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=30, R2: 61.73%
# (113, 16)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.000, n=16, R2: 61.73%
# (113, 15)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.002, n=15, R2: 61.73%
# (113, 14)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result :
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.003, n=14, R2: 61.73%
# (113, 13)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29667       validation_0-mae:0.29129     
#         validation_1-rmse:0.33484       validation_1-mae:0.31704     
# [2]     validation_0-rmse:0.24154       validation_0-mae:0.23010     
#         validation_1-rmse:0.29805       validation_1-mae:0.26303     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.296668, 0.241542], 'mae': [0.375818, 0.291286, 0.230099]}, 'validation_1': {'rmse': [0.396096, 0.334837, 0.298051], 'mae': [0.390092, 0.317043, 0.263032]}}
# Thresh=0.006, n=13, R2: 61.73%
# (113, 12)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29709       validation_0-mae:0.29126     
#         validation_1-rmse:0.33416       validation_1-mae:0.31562     
# [2]     validation_0-rmse:0.24256       validation_0-mae:0.23031     
#         validation_1-rmse:0.29807       validation_1-mae:0.26196     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297093, 0.242562], 'mae': [0.375818, 0.291265, 0.230306]}, 'validation_1': {'rmse': [0.396096, 0.334158, 0.298066], 'mae': [0.390092, 0.315617, 0.261962]}}
# Thresh=0.007, n=12, R2: 61.73%
# (113, 11)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29709       validation_0-mae:0.29126     
#         validation_1-rmse:0.33416       validation_1-mae:0.31562     
# [2]     validation_0-rmse:0.24256       validation_0-mae:0.23031     
#         validation_1-rmse:0.29807       validation_1-mae:0.26196     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297093, 0.242562], 'mae': [0.375818, 0.291265, 0.230306]}, 'validation_1': {'rmse': [0.396096, 0.334158, 0.298066], 'mae': [0.390092, 0.315617, 0.261962]}}
# Thresh=0.009, n=11, R2: 61.73%
# (113, 10)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29709       validation_0-mae:0.29126     
#         validation_1-rmse:0.33416       validation_1-mae:0.31562     
# [2]     validation_0-rmse:0.24256       validation_0-mae:0.23031     
#         validation_1-rmse:0.29807       validation_1-mae:0.26196     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297093, 0.242562], 'mae': [0.375818, 0.291265, 0.230306]}, 'validation_1': {'rmse': [0.396096, 0.334158, 0.298066], 'mae': [0.390092, 0.315617, 0.261962]}}
# Thresh=0.010, n=10, R2: 61.73%
# (113, 9)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result :
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.011, n=9, R2: 61.73%
# (113, 8)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.015, n=8, R2: 61.73%
# (113, 7)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.015, n=7, R2: 61.73%
# (113, 6)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.025, n=6, R2: 61.73%
# (113, 5)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result : 
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.027, n=5, R2: 61.73%
# (113, 4)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result :
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.045, n=4, R2: 61.73%
# (113, 3)
# [0]     validation_0-rmse:0.37780       validation_0-mae:0.37582     
#         validation_1-rmse:0.39610       validation_1-mae:0.39009     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.29758       validation_0-mae:0.29120     
#         validation_1-rmse:0.33488       validation_1-mae:0.31561     
# [2]     validation_0-rmse:0.24382       validation_0-mae:0.23031     
#         validation_1-rmse:0.29897       validation_1-mae:0.26177     
# evals_result :
#  {'validation_0': {'rmse': [0.377804, 0.297584, 0.243817], 'mae': [0.375818, 0.291198, 0.230313]}, 'validation_1': {'rmse': [0.396096, 0.334879, 0.298972], 'mae': [0.390092, 0.315609, 0.261772]}}
# Thresh=0.046, n=3, R2: 61.73%
# (113, 2)
# [0]     validation_0-rmse:0.38141       validation_0-mae:0.37898     
#         validation_1-rmse:0.40019       validation_1-mae:0.39376     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.30301       validation_0-mae:0.29469     
#         validation_1-rmse:0.34084       validation_1-mae:0.31940     
# [2]     validation_0-rmse:0.25124       validation_0-mae:0.23417     
#         validation_1-rmse:0.30754       validation_1-mae:0.26604     
# evals_result :
#  {'validation_0': {'rmse': [0.381413, 0.303014, 0.251236], 'mae': [0.378981, 0.294694, 0.234165]}, 'validation_1': {'rmse': [0.400191, 0.34084, 0.307536], 'mae': [0.393761, 0.319403, 0.266043]}}
# Thresh=0.257, n=2, R2: 60.79%
# (113, 1)
# [0]     validation_0-rmse:0.38316       validation_0-mae:0.38060     
#         validation_1-rmse:0.40190       validation_1-mae:0.39560     
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.

# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:0.30698       validation_0-mae:0.29771     
#         validation_1-rmse:0.34478       validation_1-mae:0.32316     
# [2]     validation_0-rmse:0.25703       validation_0-mae:0.23809     
#         validation_1-rmse:0.31259       validation_1-mae:0.27108     
# evals_result :
#  {'validation_0': {'rmse': [0.383163, 0.306978, 0.257025], 'mae': [0.380597, 0.297708, 0.238094]}, 'validation_1': {'rmse': [0.401902, 0.344777, 0.312587], 'mae': [0.395601, 0.323161, 0.271084]}}
# Thresh=0.521, n=1, R2: 58.93%
