# breast cancer 유방암. 이진분류

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#1. 데이터
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target
# print(y[0])         # 0,1 둘 중 하나
# print(x[0])         # 30개 컬럼
# print(x.shape)      # (569,30)
# print(y.shape)      # (569, )

#1-1. 데이터 전처리
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)      # (569,30)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=77, train_size=0.8)

#2. 모델 구성
# model = SVC()                            
# score:  0.9649122807017544
# acc:  0.9649122807017544

# model = LinearSVC()                   # 규제에 편향을 포함싴켜서 SVC보다 낮은 정확도                  
# score:  0.9649122807017544
# acc:  0.9649122807017544

# model = RandomForestClassifier()         
# score:  0.9298245614035088
# acc:  0.9298245614035088

# model = RandomForestRegressor()          
# error

model = KNeighborsClassifier()           
# score:  0.9649122807017544
# acc:  0.9649122807017544

# model = KNeighborsRegressor()            
# error

#3. 컴파일, 훈련
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=5)
# model.evaluate(딥러닝)==model.score(머신러닝)
score = model.score(x_test, y_test)
acc = accuracy_score(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

print("score: ", score)
print("acc: ", acc)
# print("r2: ", r2)
