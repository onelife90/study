# boston 회귀

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#1. 데이터
boston = load_boston()
x = boston.data
y = boston.target
# print(y)
# print(x.shape)        # (506, 13)
# print(y.shape)      # (506,)

#1-1. 데이터 전처리
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x[1])
# print(x.shape)                  # (506, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

#2. 모델 구성
# model = SVC()                           
# ValueError: Unknown label type: 'continuous'

# model = LinearSVC()                     
# ValueError: Unknown label type: 'continuous'  

# model = RandomForestClassifier()        
# ValueError: Unknown label type: 'continuous'

# model = RandomForestRegressor()         
# score:  0.8585668166937119
# R2:  0.8585668166937119

# model = KNeighborsClassifier()          
# ValueError: Unknown label type: 'continuous'

model = KNeighborsRegressor()           
# score:  0.7510522283616072
# R2:  0.7510522283616072

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#4. 평가, 예측
# loss, mse = model.evaluate(x_test, y_test, batch_size=1)
score = model.score(x_test, y_test)
# acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("score: ", score)         
# print("acc: ", acc)         
print("R2: ", r2)         
