# 교차검증의 분할수를 5로 지정하여 출력하세요

from sklearn import svm, datasets, cross_validation

iris = datasets.load_iris()
x = iris.data
y = iris.target

# 머신러닝 알고리즘 SVM 사용
svc = svm.SVC(c=1, kernel='rbf', gamma=0.001)

# 내부에서는 x_train, x_test, y_train, y_test로 분할 처리
scores = cross_validation.cross_val_score(svc, x, y, cv=5)

print(scores)
print("평균점수: ", scores.mean())

# ImportError: cannot import name 'cross_validation' from 'sklearn'
