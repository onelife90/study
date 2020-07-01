# 보스턴 모델링 하시오
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import warnings

# warnigs라는 에러를 그냥 넘어가겠다
warnings.filterwarnings('ignore')

#1-1. csv 파일 불러오기
boston = pd.read_csv('./data/csv/boston_house_prices.csv', header=1)
# boston.info() # dtypes: float64(11), int64(3)

x = boston.iloc[:, 0:13]    
y = boston.iloc[:, 13]
# print(x)    # [506 rows x 13 columns]
# print(y)    # Name: MEDV, Length: 506, dtype: float64

#1-2. train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, test_size=0.2)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='regressor')
# all_estimator에 사이킷런의 모든 회귀 모델이 저장

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 = ", r2_score(y_test, y_pred))
# R2 값이 음수가 될 수 있나?
# 이러한 R^2의 단점은 데이터가 매우 단순하여 평균을 예측하는 것만으로도 오차가 충분히 작을 때에는 모델 성능이 좋든 나쁘든 유사한 지표가 측정될 수 있다는 점일 것이다.
# 이때에는 편차가 매우 작아지고, 오차도 동시에 매우 작아지게 된다. 그러면 경우에 따라서 우측항이 1보다 큰 값을 낼 수 있어 R^2가 0 이하가 될 수 있다.

import sklearn
print(sklearn.__version__)
# ARDRegression 의 정답률 =  0.6496265879106574
# AdaBoostRegressor 의 정답률 =  0.7501731154942928
# BaggingRegressor 의 정답률 =  0.8009785138964878
# BayesianRidge 의 정답률 =  0.6437947120234067
# CCA 의 정답률 =  0.5757371047671898
# DecisionTreeRegressor 의 정답률 =  0.6865359707460095
# ElasticNet 의 정답률 =  0.6136260501601751
# ElasticNetCV 의 정답률 =  0.6059369251614932
# ExtraTreeRegressor 의 정답률 =  0.6315501680406792
# ExtraTreesRegressor 의 정답률 =  0.7938383185841877
# GaussianProcessRegressor 의 정답률 =  -5.6887784116555435
# GradientBoostingRegressor 의 정답률 =  0.8214816923636068
# HuberRegressor 의 정답률 =  0.5698309394868109
# KNeighborsRegressor 의 정답률 =  0.5850720309992157
# KernelRidge 의 정답률 =  0.6352634330733764
# Lars 의 정답률 =  0.624926999450356
# LarsCV 의 정답률 =  0.624926999450356
# Lasso 의 정답률 =  0.6124106278092214
# LassoCV 의 정답률 =  0.6241218817677056
# LassoLars 의 정답률 =  -0.012439932990852887
# LassoLarsCV 의 정답률 =  0.6674690355194659
# LassoLarsIC 의 정답률 =  0.666264686991979
# LinearRegression 의 정답률 =  0.6674690355194655
# LinearSVR 의 정답률 =  0.5798322749527361
# MLPRegressor 의 정답률 =  0.505651704727687
