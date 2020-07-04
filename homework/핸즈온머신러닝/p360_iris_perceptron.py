import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:, (2,3)] # 꽃잎의 길이와 너비
y = (iris.target==0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)

# 퍼셉트론 학습 알고리즘은 확률적 경사 하강법과 매우 닮았다
# Perceptron의 매개변수 loss='perceptron', learning_rate="constant", eta0=1(학습률), penalty=None인 SGDClassifier과 같다

# 퍼셉트론을 여러 개 쌓아올리면 일부 제약을 줄일 수 있다 ==> 다층 퍼셉트론(MLP)
