from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,0,0,1]
# and_|___0___1_____
#     |
#  0  |   0   0      
#     |
#  1  |   0   1
#     |

#2. 모델
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_pred = model.predict(x_test)

# score = model.evaluate(예측)
acc = accuracy_score([0,0,0,1], y_pred)
print(x_test, "의 예측 결과: ", y_pred)
print("acc = ", acc)
