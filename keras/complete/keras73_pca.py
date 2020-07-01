# CSV 데이터 파일. 엑셀 친구. 행렬로 되어있음
# 받아서 Pandas(여러가지 데이터 형식), Numpy(한가지 데이터 형식)로 변환
# 데이터에는 결측치, 이상치가 포함
# 결측치 제거 방법 : train_test_split해서 predict 값을 얻고 메운다
# 이상치 제거 방법 : robustscaler(중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환)을 써서 이상치 모두 날려버린다

# PCA : 중요한 데이터를 압축하는 방법. 머신러닝 기법. 차원(컬럼)축소
# 주성분 분석을 해서 압축 진행 cf)Maxpooling
# PCA에서는 잠재변수와 측정 데이터가 선형적인 관계로 연결되어 있다고 가정
# from sklearn.decomposition import PCA
# pca1 = PCA(n_components=1)
# X_low = pca1.fit_transform(X)         # 낮은 차원 변환
# X2 = pca1.inverse_transform(X_low)    # 원래 차원 복귀
