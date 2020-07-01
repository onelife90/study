# pipeline : 전처리도 한 방에 처리하는 자동화

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=88)

#2. 모델 구성
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
# SVC 모델과 MinMaxScaler를 쓰겠다
pipe = make_pipeline(MinMaxScaler(), SVC())

#3. 훈련
# 전처리와 모델을 한꺼번에 처리하는 것이 pipe이기 때문에 pipe.fit
pipe.fit(x_train, y_train)
print("acc: ", pipe.score(x_test, y_test))
# acc:  0.9666666666666667
