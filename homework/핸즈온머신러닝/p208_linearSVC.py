import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
# Pipeline : 연속된 변환을 순서대로 처리하게 도와주는 클래스
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
x = iris["data"][:, (2,3)]      # 꽃의 길이, 꽃잎 너비
y = (iris["target"]==2).astype(np.float64)  # Iris-Virginica

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])

svm_clf.fit(x,y)
