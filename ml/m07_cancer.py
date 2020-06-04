# breast cancer 유방암. 이진분류

import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#1. 데이터
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target
# print(y[0])         # 0,1 둘 중 하나
# print(x[0])         # 30개 컬럼
# print(x.shape)      # (569,30)
# print(y.shape)      # (569, )

#1-1. 데이터 전처리
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)      # (569,30)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, train_size=0.6)

#2. 모델 구성
# model = SVC()                            
# score:  0.9868421052631579
# acc:  0.9868421052631579
# r2:  0.9434523809523809

# model = LinearSVC()                      
# score:  0.9692982456140351
# acc:  0.9692982456140351
# r2:  0.8680555555555556

# model = RandomForestClassifier()         
# score:  0.956140350877193
# acc:  0.956140350877193
# r2:  0.8115079365079365

# model = RandomForestRegressor()          
# error

# model = KNeighborsClassifier()           
# score:  0.9649122807017544
# acc:  0.9649122807017544
# r2:  0.8492063492063492

model = KNeighborsRegressor()            
# error

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=5)
score = model.score(x_test, y_test)
acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("score: ", score)
print("acc: ", acc)
print("r2: ", r2)
# print("loss: ", loss)
# print("acc: ", acc)
