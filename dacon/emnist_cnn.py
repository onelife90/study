import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

#1. 데이터 load
train = pd.read_csv('./data/dacon/emnist/train.csv', header=0, index_col=0, sep=',')
test = pd.read_csv('./data/dacon/emnist/test.csv', header=0, index_col=0, sep=',')
submit = pd.read_csv('./data/dacon/emnist/submission.csv', header=0, index_col=0, sep=',')

#1-1. train min, max값 출력
# print(f'train_row_max: \n {train.max(axis=1)}')
# train_col_max:
#  id
# 1       249
# 2       255
# 3       255
# 4       255
# 5       255
#        ...
# 2044    253
# 2045    255
# 2046    255
# 2047    251
# 2048    255
# Length: 2048, dtype: int64

# print()

# print(f'train_row_max: \n {train.max(axis=0)}')
# train_row_max:
#  digit     9
# letter    Z
# 0         4
# 1         4
# 2         4
#          ..
# 779       4
# 780       4
# 781       4
# 782       4
# 783       4
# Length: 786, dtype: object

#1-2. 픽셀 컬럼 정규화
train_pixel = train.iloc[:,2:]/255
# print(train_pixel)
test_pixel = test.iloc[:,1:]/255
# print(test_pixel)

#1-3. letter 컬럼 원핫인코딩
train_letter = train.iloc[:,1]
train_letter = pd.get_dummies(train_letter, columns=['letter'])
# print(f'train_letter:\n{train_letter}')

test_letter = test.iloc[:,0]
test_letter = pd.get_dummies(test_letter, columns=['letter'])
# print(f'test_letter:\n{test_letter}')

#1-4. 픽셀 컬럼 + letter 컬럼 병합
x_train = np.concatenate()


'''
train = pd.get_dummies(train.iloc[:,:1])#, columns=['letter'])
print(f'train:\n{train}')   # [2048 rows x 811 columns]

test = pd.get_dummies(test, columns=['letter'])
print(f'test:\n{test}')     # [20480 rows x 810 columns]


y_train = train['digit']
train = train.drop(['digit'], axis=1)
# print(f'train:\n{train}')   # [2048 rows x 810 columns]


#1-3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(train, y_train, train_size=0.8, random_state=99)
# print(f'x_train,x_test: {x_train.shape} {x_test.shape}')    # x_train,x_test: (1638, 810) (410, 810)
# print(f'y_train,y_test: {y_train.shape} {y_test.shape}')    # x_train,x_test: (1638, ) (410, )

#1-4. cnn_input shape 맞추기
x_train = x_train.reshape(-1,)

#1-4. 모델 구성



#1-5. 훈련
model.fit(x_train,y_train)

#1-6. 평가, 예측
'''
