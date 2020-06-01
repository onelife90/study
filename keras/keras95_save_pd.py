# csv 파일 저장 경로를 알고싶으면? print(data명)
# iris.csv 파일에 필요없는 헤더가 있으므로 제거해주자
import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", index_col=None, header=0, sep=',')
# index_col=None 읽기 전 파일에 index_col에 데이터가 껴있었다. 그래서 None
# header=0을 하면 첫 헤더(행)는 실 데이터로 인식 X
print(datasets)

print(datasets.head())      # 위에서부터 5개
print(datasets.tail())      # 아래서부터 5개

print("======================")
print(datasets.values)      # ★항상 쓰임. 머신을 돌리기 위해서 np로 변환

aaa = datasets.values
print(type(aaa))            # <class 'numpy.ndarray'>

# np로 저장하시오

from sklearn.datasets import load_iris

iris = load_iris()

np.save('./data/iris.npy', arr=iris)
