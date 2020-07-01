import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
# print(type(iris))       # <class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target
# print(type(x_data))     # <class 'numpy.ndarray'>
# print(type(y_data))     # <class 'numpy.ndarray'>

# save 할 때는 arr(배열 형태로 저장, 변수명)
np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)

# load 할 때는 경로만 입력
x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load))        # <class 'numpy.ndarray'>
print(type(y_data_load))        # <class 'numpy.ndarray'>
print(x_data_load.shape)        # (150,4)
print(y_data_load.shape)        # (150, )
