# 손실값이 있다고해도 acc가 0.99가 나올 수 있을까?
# 있음. mnist에서는 0(쓰레기값)이 배경으로 많이 차지하고 있기 때문에 그 정도의 손실율이어도 acc가 잘 나올 수 있음

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

#1. 데이터
dataset = load_diabetes()
X = dataset.data
Y = dataset.target

# print(f'X.shape: {X.shape}')    # X.shape: (442, 10)
# print(f'Y.shape: {Y.shape}')    # Y.shape: (442,)

#1-1. 데이터 전처리
# pca = PCA(n_components=5)
# x2 = pca.fit_transform(X)
# pca_evr = pca.explained_variance_ratio_
# # pca한 각각의 주성분 벡터의 축소 비율을 보여줌

# print(f'pca_evr : {pca_evr}')
# # pca_evr : [0.40242142 0.149282 0.1259623 0.09554764 0.06621856]
# print(f'sum(pca_evr): {sum(pca_evr)}')
# # sum(pca_evr): 0.8340156689459766 손실값이 있음

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수

print(f'cumsum: {cumsum}')
#cumsum: [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.]

n_components = np.argmax(cumsum>=0.94)+1
# 1을 더해주는 이유는 인덱스가 0으로 시작하기 때문에 n_components가 7번째가 되어야 하기 때문에

print(f'type(cumsum): {type(cumsum)}')
# type(cumsum): <class 'numpy.ndarray'>

print(f'cumsum>=0.94: {cumsum>=0.94}')
# 타입이 ndarray인 cumsum의 각 요소가 0.94 이상인 것인지 묻는 조건문이기 때문에 Ture/False로 출력
#cumsum>=0.94: [False False False False False False  True  True  True  True]

print(f'n_components: {n_components}')
