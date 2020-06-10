import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
model = RandomForestRegressor(n_estimators=50)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_pred, y_test)
submit = model.predict(x_pred)

print("score: ", score)
print("MAE: ", mae)
# score:  0.19545216613503955
# MAE:  1.5719956749999986

#5. submit할 파일 생성
submit = pd.DataFrame(submit, index=np.arange(10000,20000))
submit.to_csv('./data/dacon/comp1/submission_RF.csv', header=["hhb","hbo2","ca","na"], index=True, index_label="id")
