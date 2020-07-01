# xor 모델을 완성하시오.
# 다른 방법

from sklearn.svm import SVC
# 서포트벡터 : 아웃풋 중에서 가장 경계선에 가까이 붙어있는 최전방의 데이터들
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# KNeighbors 최근접 이웃 / 데이터 별로 근접한 기준을 사용

#1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]
# xor_|___0___1_____
#     |
#  0  |   0   1      
#     |
#  1  |   1   0
#     |

#2. 모델
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier(n_neighbors=1)
# n_neighbors=2로 하게 되면 acc 0.5 / 1번째 이웃한테 가고 2번째 이웃일때 자기 자신한테 돌아오는듯

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_pred = model.predict(x_test)

# score = model.evaluate(예측)
acc = accuracy_score([0,1,1,0], y_pred)
print(x_test, "의 예측 결과: ", y_pred)
print("acc = ", acc)
# acc =  1.0
