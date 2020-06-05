# wine의 y값이 편중되어 있는 문제!
# 이 문제를 어떻게 해결할까?
# 인위적으로 이 편차의 폭을 축소하면 조금 균등한 분포가 된다

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#1-1. 와인 데이터 읽기
wine = pd.read_csv('./data/winequality-white.csv', sep=';', header=0, index_col=None)

y = wine['quality']
x = wine.drop('quality', axis=1)
# wine이라는 데이터에서 quality라는 컬럼을 drop 하겠다
# print(x.shape)  # (4898, 11)
# print(y.shape)  # (4898, )

#1-2. y 레이블 축소(y값을 0,1,2로 세가지 분류)
newlist = []
for i in list(y):
    if i <= 4:
        newlist +=[0]
    elif i <= 7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# 모델 구성
model = RandomForestClassifier()

# 훈련
model.fit(x_train, y_train)

# 평가, 예측
y_pred = model.predict(x_test)
acc = model.score(x_test, y_test)
print("acc_score: ", accuracy_score(y_test, y_pred))
print("acc: ", acc)
# acc_score:  0.9448979591836735
# acc:  0.9448979591836735
