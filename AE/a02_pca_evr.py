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
pca = PCA(n_components=5)
x2 = pca.fit_transform(X)
pca_evr = pca.explained_variance_ratio_
# pca한 각각의 주성분 벡터의 축소 비율을 보여줌

# print(f'pca_evr : {pca_evr}')
# pca_evr : [0.40242142 0.149282 0.1259623 0.09554764 0.06621856]
# print(f'sum(pca_evr): {sum(pca_evr)}')
# sum(pca_evr): 0.8340156689459766 손실값이 있음
