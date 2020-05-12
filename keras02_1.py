# 데이터 구성
from keras.models impport Sequential
from keras.layers import Dense
import numpy as np

# 데이터 생성
x_train = np.array([1,3,5,7,9,11,13,15,17,19])
y_train = np.arraay([11,33,55,77,99,121,143,165,187,209])
x_test = np.array([2,4,6,8,10,12,14,16,18,20])
y_test = np.array([22,44,66,88,110,132,154,176,198,220)

# 모델 구성
model = Sequential()
model.add(Dense(5), input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

model.summary

# 훈련
model.compile(loss='mse', optimizer='adam', metricks=['accuracy'])
model.fit(x_train, y_train, ephocs=100, batch_size=1,
            validation_data= (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

# 예측
print("loss : ", loss)
print("acc : ", acc)

                   
#하이퍼파라미터
                   
#epochs=100, batch_size=1, Dense=5,3,1
#loss : 14(1) / 105(2) / 18634(3)
#acc : 0.1(1) / 0.0(2) / 0.0(3)
#안깊어서?
