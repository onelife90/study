# 그라디언트 부스터. 랜덤포레스트 친구
# RF. 숲이 있으면 나무가 있다
# 결정트리 모델. 결정나무. DecisionTree 여러개가 모여있으면 앙상블
# from sklearn.ensemble import GradientBoostingClassifier
# 트리 구조 model에서 주요 parameter는 1)max_depth : 깊이 2)feature_importance : 컬럼별 중요도 
# 사용법 : print(model.feature_importances_)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.8, random_state=42)
# print(x_train.shape)    # (455, 30)
# print(x_test.shape)     # (114, 30)
# print(y_train.shape)    # (455, )
# print(y_test.shape)     # (114, )

#2. 모델 구성
model = GradientBoostingClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
acc = model.score(x_test, y_test)
print("acc: ", acc)
print(model.feature_importances_)
# 30개 출력(30개의 숫자 중에 큰 숫자가 가장 큰 영향을 주는 컬럼이다)
# 대회 나가서 PCA를 먼저 돌리고 model.feature_importances_를 실행시키면 빠른 속도로 결정 가능
# 트리 구조이기 때문에 전처리가 필요 없지만 과적합의 우려(max_depth=5 이상)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    # 가로방향으로 바 차트를 그릴 시 barh 명령. align=bar의 정렬 위치
    plt.yticks(np.arange(n_features), cancer.feature_names)
    # ticker의 위치와 각 위치에서의 label 설정
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    # ylim y축의 최소값과 최대값을 지정
plot_feature_importance_cancer(model)
plt.show()
