from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train, y_train, x_test, y_test의 크기를 출력
print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(x_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000,)
# train 데이터 6만, test 데이터 1만

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)도 가능
