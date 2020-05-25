# 변수 confmat에 y_true와 y_pred의 혼동행렬을 저장하세요

# 혼동행렬의 구성
# 참양성   | 거짓음성
# 거짓양성 | 참음성

import numpy
from sklearn.metrics import confusion_matrix

# 0=양성, 1=음성
y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]

# 참양성(0,0)==2개   | 거짓음성(0,1)==1개
# 거짓양성(1,0)==0개 | 참음성(1,1)==3개

confmat = confusion_matrix(y_true, y_pred)

print(confmat)
# [[2 1]
#  [0 3]]
