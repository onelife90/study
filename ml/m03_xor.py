# xor 모델을 완성하시오.
# SVC import

from sklearn.svm import LinearSVC
# 페널티와 로스 기능에서 좀 더 유용하며 많은 데이터에 강하다
# l2 라는 페널티가 디폴트
from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]
# xor_|___0___1_____
#     |
#  0  |   0   1      
#     |
#  1  |   1   0
#     |

# 인공지능의 겨울이 찾아옴. 어떻게 선형분리 할것인가?
# 실습. acc1.0이 나오는 방법을 찾으시오

#2. 모델
model = LinearSVC()
# acc =  0.25
# model = LinearSVR(C=1, dual=True, epsilon=0.5)

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_pred = model.predict(x_test)

# score = model.evaluate(예측)
acc = accuracy_score([0,1,1,1], y_pred)
print(x_test, "의 예측 결과: ", y_pred)
print("acc = ", acc)
