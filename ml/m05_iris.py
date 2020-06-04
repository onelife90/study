# 분류모델과 회귀모델을 각각 완성하시오
# iris 다중분류
# LogisticRegression ==> 분류
# 속도가 빠른 러신머닝. 결측치를 x_pred로 넣은 다음 나머지를 머신 돌림
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
# print(x[0])         # [5.1 3.5 1.4 0.2] == 4컬럼
# print(y)            # 0,1,2 세 종류
# print(x.shape)      # (150, 4)
# print(y.shape)      # (150, )

#1-1. 데이터 전처리
scaler = StandardScaler()
scaler.fit_transform(x)

#1-2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, train_size=0.6, shuffle=True)
# print(x_train.shape)        # (90, 4)
# print(x_test.shape)         # (60, 4)
# print(y_train.shape)        # (90, 3)
# print(y_test.shape)         # (60, 3)

#2. 모델 구성
# model = SVC()                           
# score:  0.95
# acc:  0.95
# R2 :  0.9204946996466431

# model = LinearSVC()                     
# score:  0.9666666666666667
# acc:  0.9666666666666667
# R2 :  0.9469964664310954

# model = RandomForestClassifier()
# score:  0.95
# acc:  0.95
# R2 :  0.9204946996466431

# model = RandomForestRegressor()         
# error

# model = KNeighborsClassifier()          
# score:  0.9666666666666667
# acc:  0.9666666666666667
# R2 :  0.9469964664310954

model = KNeighborsRegressor()           
# error

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
score = model.score(x_test, y_test)
acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# model.score 분류, 회귀 둘 다 사용 가능
print("score: ", score)         
print("acc: ", acc)         
print("R2 : ", r2)

# print("loss: ", loss)
# print("acc: ", acc)
