# DecisionTree, RF, GB, XGB로 feature_importance 적용하기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

#1. csv 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

#1-1. 데이터 결측치 제거
# print(train.isnull().sum())
# train이 있는 데이터에 null값의 합계를 가져와라
train = train.interpolate()
train = train.fillna(train.mean())
# 컬럼별 보간법. 선형보간 (평타 85점) 옆에 컬럼에 영향 X
# 빈자리를 선에 맞게 그려준다
# 선형의 시작점이 nan이면 결측지 제거 X
# print(train.isnull().sum())
test = test.interpolate()
test = test.fillna(test.mean())
x_col = train.iloc[:,:71]

#1-2. 넘파이 저장
train = train.values
test = test.values
submission = submission.values
np.save('./data/dacon/comp1/train.npy', arr=train)
np.save('./data/dacon/comp1/test.npy', arr=test)
np.save('./data/dacon/comp1/submission.npy', arr=submission)

#1-3. 넘파이 불러오기
train = np.load('./data/dacon/comp1/train.npy')
test = np.load('./data/dacon/comp1/test.npy')
submission = np.load('./data/dacon/comp1/submission.npy')

#1-2. 데이터 자르기
x = train[:, :71]
y = train[:, -4:]
# print(x.shape)  # (10000, 71)
x_pred = test
# print(x_pred.shape) # (10000, 71)

#1-3. Scaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_pred = scaler.fit_transform(x_pred)

#1-6. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=99)
# print(x_train.shape)    # (8000, 71)
# print(x_test.shape)     # (2000, 71)
# print(y_train.shape)    # (8000, 4)
# print(y_test.shape)     # (2000, 4)

#2. 모델 구성
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
score = model.score(x_test, y_test)
print("score: ", score)

#feature_importances_산출
def plot_feature_importance_(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    # 가로방향으로 바 차트를 그릴 시 barh 명령. align=bar의 정렬 위치
    plt.yticks(np.arange(n_features), x_col)
    # ticker의 위치와 각 위치에서의 label 설정
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    # ylim y축의 최소값과 최대값을 지정
plot_feature_importance_(model)
plt.show()

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_pred, y_test)
print("MAE: ", mae)

submit = model.predict(x_pred)

# DecisionTree
# score:  0.048912362965680266
# MAE:  1.7209961010173156

# RF
# score:  0.20671395762473016
# MAE:  1.56552955

# GradientBoostingRegressor
# error

# XGBRegressor
# error

# #5. submit할 파일 생성
# submit = pd.DataFrame(submit, index=np.arange(10000,20000))
# submit.to_csv('./data/dacon/comp1/submission_RF.csv', header=["hhb","hbo2","ca","na"], index=True, index_label="id")
