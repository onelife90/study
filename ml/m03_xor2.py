# xor 모델을 완성하시오.
# SVC import

from sklearn.svm import SVC
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

#2. 모델
# model = LinearSVC(penalty='l1', loss='squared_hinge', dual=True)
model = SVC()
# SVC 모델 사용. 서포트 벡터 클래스 (Support Vector Classifier) 클래스
# 경계선을 두개 그어주어 서포트벡터와 경계선 사이의 거리를 최소화

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
