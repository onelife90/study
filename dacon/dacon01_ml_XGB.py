# DecisionTree, RF, GB, XGB로 feature_importance 적용하기
# GB, XGB 모델은 아웃풋이 1개가 되어야 작동!
# 순서
#1) y 데이터를 PCA(n=1)로 차원축소
#2) train/test 분리
#3) 모델 구성
#4) 훈련, 평가, 예측
#5) 제출용 새로운 변수명 생성, 2차원으로 reshape
#6) 제출용 y데이터 차원복귀 pca.inverse_transform(n=4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

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

#1-4. 데이터 자르기
x = train[:, :71]
y = train[:, -4:]
# print(x.shape)  # (10000, 71)
x_pred = test
# print(x_pred.shape) # (10000, 71)

#1-5. PCA, scaler
pca = PCA(n_components=1)
y = pca.fit_transform(y)
# print(y)
# print(y.shape)      # (10000, 1)

# scaler
scaler = StandardScaler()
y = scaler.fit_transform(y)

#1-6. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=20)
# print(x_train.shape)    # (8000, 71)
# print(x_test.shape)     # (2000, 71)
# print(y_train.shape)    # (8000, 1)
# print(y_test.shape)     # (2000, 1)

#2. 모델 구성
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

# MAE 평가 지표에 넣기 위한 y_pred
y_pred = model.predict(x_test)
# print(y_pred.shape)     # (2000,)
mae = mean_absolute_error(y_pred, y_test)
print("MAE: ", mae)

# 제출할 y데이터 2차원 reshape
# PCA를 거친 y는 벡터화가 되므로 reshape
submit = model.predict(x_pred)
# print(submit.shape)     # (10000,)
submit = submit.reshape(-1,1)
# print(submit.shape)     # (10000,1)

# 제출할 때는 y 4컬럼_PCA.inverse_transform
submit = pca.inverse_transform(submit)
# print(submit.shape)    # (10000, 4)
print(submit)   

# XGBRegressor
# score:  0.4936095667826256
# MAE:  0.5740817345786561

#5. submit할 파일 생성
submit = pd.DataFrame(submit, index=np.arange(10000,20000))
submit.to_csv('./data/dacon/comp1/submission_XGB.csv', header=["hhb","hbo2","ca","na"], index=True, index_label="id")
