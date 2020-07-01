# xor 모델을 완성하시오.
# 선형으로 분류하겠다

from sklearn.svm import LinearSVC
# 선형 분류 알고리즘 support vector Calssifier
# 페널티와 로스 기능에서 좀 더 유용하며 많은 데이터에 강하다
# 제곱힌지 손실함수, l2 규제(디폴트)
# 규제의 강도를 결정하는 매개변수 C
# C 값이 높아지면 규제 감소. Card 한도를 높게하여 규제를 풀어주는 거라 생각하자
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

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_pred = model.predict(x_test)

# score = model.evaluate(예측)
acc = accuracy_score([0,1,1,0], y_pred)
print(x_test, "의 예측 결과: ", y_pred)
print("acc = ", acc)
# acc =  0.5
