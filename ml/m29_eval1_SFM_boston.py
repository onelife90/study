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
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

#3-1. 컬럼수 만큼 돌 thresholds 생성
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00284344 0.00317365 0.00332973 0.00815478 0.00844738 0.01070327   
#  0.01749661 0.04932566 0.05530359 0.06542572 0.08799212 0.307893     
#  0.379911  ]

#3-2. SelectFromModel 생성
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape)

    selection_model = XGBRegressor(n_estimators=3, n_jobs=-1) 

    selection_model.fit(selection_x_train, y_train, verbose=True, eval_metric=["rmse", "mae"], 
                        eval_set=[(selection_x_train, y_train), (selection_x_test, y_test)], early_stopping_rounds=3)
    
    y_pred = selection_model.predict(selection_x_test)
    
    results = selection_model.evals_result()
    print("evals_result : \n", results)
    
    score = r2_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))

# (101, 13)
# [0]     validation_0-rmse:16.83263      validation_0-mae:15.11649    
#         validation_1-rmse:17.95198      validation_1-mae:16.14069    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.34330      validation_0-mae:10.81502    
#         validation_1-rmse:13.70303      validation_1-mae:11.84140    
# [2]     validation_0-rmse:9.16812       validation_0-mae:7.72275     
#         validation_1-rmse:10.59323      validation_1-mae:8.81333     
# evals_result : 
#  {'validation_0': {'rmse': [16.832626, 12.3433, 9.168118], 'mae': [15.11649, 10.815018, 7.722749]}, 
# 'validation_1': {'rmse': [17.951984, 13.703028, 10.593233], 'mae': [16.140686, 11.841396, 8.813331]}}      
# Thresh=0.003, n=13, R2: -33.38%

# (101, 12)
# [0]     validation_0-rmse:16.83263      validation_0-mae:15.11649    
#         validation_1-rmse:17.95198      validation_1-mae:16.14069    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.34330      validation_0-mae:10.81502    
#         validation_1-rmse:13.70303      validation_1-mae:11.84140    
# [2]     validation_0-rmse:9.16812       validation_0-mae:7.72275     
#         validation_1-rmse:10.59323      validation_1-mae:8.81333     
# evals_result : 
#  {'validation_0': {'rmse': [16.832626, 12.3433, 9.168118], 'mae': [15.11649, 10.815018, 7.722749]}, 
# 'validation_1': {'rmse': [17.951984, 13.703028, 10.593233], 'mae': [16.140686, 11.841396, 8.813331]}}      
# Thresh=0.003, n=12, R2: -33.38%

# (101, 11)
# [0]     validation_0-rmse:16.83263      validation_0-mae:15.11649    
#         validation_1-rmse:17.95198      validation_1-mae:16.14069    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.34330      validation_0-mae:10.81502    
#         validation_1-rmse:13.70303      validation_1-mae:11.84140    
# [2]     validation_0-rmse:9.16812       validation_0-mae:7.72275     
#         validation_1-rmse:10.59323      validation_1-mae:8.81333     
# evals_result : 
#  {'validation_0': {'rmse': [16.832626, 12.3433, 9.168118], 'mae': [15.11649, 10.815018, 7.722749]}, 'validation_1': {'rmse': [17.951984, 13.703028, 10.593233], 'mae': [16.140686, 11.841396, 8.813331]}}      
# Thresh=0.003, n=11, R2: -33.38%

# (101, 10)
# [0]     validation_0-rmse:16.83263      validation_0-mae:15.11649    
#         validation_1-rmse:17.95198      validation_1-mae:16.14069    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.34330      validation_0-mae:10.81502    
#         validation_1-rmse:13.70303      validation_1-mae:11.84140    
# [2]     validation_0-rmse:9.16812       validation_0-mae:7.72275     
#         validation_1-rmse:10.59323      validation_1-mae:8.81333     
# evals_result : 
#  {'validation_0': {'rmse': [16.832626, 12.3433, 9.168118], 'mae': [15.11649, 10.815018, 7.722749]}, 'validation_1': {'rmse': [17.951984, 13.703028, 10.593233], 'mae': [16.140686, 11.841396, 8.813331]}}      
# Thresh=0.008, n=10, R2: -33.38%

# (101, 9)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:17.95692      validation_1-mae:16.09732    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.71772      validation_1-mae:11.79384    
# [2]     validation_0-rmse:9.19969       validation_0-mae:7.71456     
#         validation_1-rmse:10.60959      validation_1-mae:8.79086     
# evals_result :
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.199693], 'mae': [15.076819, 10.797745, 7.714557]}, 'validation_1': {'rmse': [17.956919, 13.717719, 10.609586], 'mae': [16.097317, 11.793838, 8.790862]}}   
# Thresh=0.008, n=9, R2: -33.79%

# (101, 8)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:17.95692      validation_1-mae:16.09732    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.71772      validation_1-mae:11.79384    
# [2]     validation_0-rmse:9.19969       validation_0-mae:7.71456     
#         validation_1-rmse:10.60959      validation_1-mae:8.79086     
# evals_result : 
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.199693], 'mae': [15.076819, 10.797745, 7.714557]}, 'validation_1': {'rmse': [17.956919, 13.717719, 10.609586], 'mae': [16.097317, 11.793838, 8.790862]}}   
# Thresh=0.011, n=8, R2: -33.79%

# (101, 7)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:18.06585      validation_1-mae:16.19108    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.83748      validation_1-mae:11.88760    
# [2]     validation_0-rmse:9.20029       validation_0-mae:7.71229     
#         validation_1-rmse:10.76588      validation_1-mae:8.91037     
# evals_result :
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.200294], 'mae': [15.076819, 10.797745, 7.712289]}, 'validation_1': {'rmse': [18.065853, 13.837481, 10.765877], 'mae': [16.191076, 11.8876, 8.910373]}}     
# Thresh=0.017, n=7, R2: -37.77%

# (101, 6)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:18.06585      validation_1-mae:16.19108    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.83748      validation_1-mae:11.88760    
# [2]     validation_0-rmse:9.20029       validation_0-mae:7.71229     
#         validation_1-rmse:10.76588      validation_1-mae:8.91037     
# evals_result : 
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.200294], 'mae': [15.076819, 10.797745, 7.712289]}, 'validation_1': {'rmse': [18.065853, 13.837481, 10.765877], 'mae': [16.191076, 11.8876, 8.910373]}}     
# Thresh=0.049, n=6, R2: -37.77%

# (101, 5)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:18.06585      validation_1-mae:16.19108    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.83748      validation_1-mae:11.88760    
# [2]     validation_0-rmse:9.20029       validation_0-mae:7.71229     
#         validation_1-rmse:10.76588      validation_1-mae:8.91037     
# evals_result : 
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.200294], 'mae': [15.076819, 10.797745, 7.712289]}, 'validation_1': {'rmse': [18.065853, 13.837481, 10.765877], 'mae': [16.191076, 11.8876, 8.910373]}}     
# Thresh=0.055, n=5, R2: -37.77%

# (101, 4)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:18.06585      validation_1-mae:16.19108    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.83748      validation_1-mae:11.88760    
# [2]     validation_0-rmse:9.20029       validation_0-mae:7.71229     
#         validation_1-rmse:10.76588      validation_1-mae:8.91037     
# evals_result :
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.200294], 'mae': [15.076819, 10.797745, 7.712289]}, 'validation_1': {'rmse': [18.065853, 13.837481, 10.765877], 'mae': [16.191076, 11.8876, 8.910373]}}     
# Thresh=0.065, n=4, R2: -37.77%

# (101, 3)
# [0]     validation_0-rmse:16.84227      validation_0-mae:15.07682    
#         validation_1-rmse:18.06585      validation_1-mae:16.19108    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.36656      validation_0-mae:10.79775    
#         validation_1-rmse:13.85305      validation_1-mae:11.89596    
# [2]     validation_0-rmse:9.21099       validation_0-mae:7.70796     
#         validation_1-rmse:10.76105      validation_1-mae:8.85407     
# evals_result :
#  {'validation_0': {'rmse': [16.842272, 12.366563, 9.210991], 'mae': [15.076819, 10.797745, 7.707962]}, 'validation_1': {'rmse': [18.065853, 13.85305, 10.761046], 'mae': [16.191076, 11.895958, 8.854074]}}    
# Thresh=0.088, n=3, R2: -37.64%

# (101, 2)
# [0]     validation_0-rmse:16.84871      validation_0-mae:15.07127    
#         validation_1-rmse:17.75928      validation_1-mae:15.91869    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.40607      validation_0-mae:10.75966    
#         validation_1-rmse:13.36783      validation_1-mae:11.54640    
# [2]     validation_0-rmse:9.27757       validation_0-mae:7.74779     
#         validation_1-rmse:10.30277      validation_1-mae:8.58082     
# evals_result :
#  {'validation_0': {'rmse': [16.848707, 12.406068, 9.277574], 'mae': [15.071272, 10.759658, 7.747793]}, 'validation_1': {'rmse': [17.759279, 13.367835, 10.30277], 'mae': [15.91869, 11.546395, 8.580816]}}     
# Thresh=0.308, n=2, R2: -26.17%

# (101, 1)
# [0]     validation_0-rmse:16.93544      validation_0-mae:15.05603    
#         validation_1-rmse:17.76201      validation_1-mae:15.87320    
# Multiple eval metrics have been passed: 'validation_1-mae' will be used for early stopping.
# Will train until validation_1-mae hasn't improved in 3 rounds.       
# [1]     validation_0-rmse:12.62328      validation_0-mae:10.78725    
#         validation_1-rmse:13.32581      validation_1-mae:11.43790    
# [2]     validation_0-rmse:9.64148       validation_0-mae:7.74444     
#         validation_1-rmse:10.33670      validation_1-mae:8.35209     
# evals_result :
#  {'validation_0': {'rmse': [16.935444, 12.623277, 9.641484], 'mae': [15.056026, 10.787246, 7.744439]}, 'validation_1': {'rmse': [17.762012, 13.325809, 10.336697], 'mae': [15.873199, 11.437897, 8.35209]}}    
# Thresh=0.380, n=1, R2: -27.00%
