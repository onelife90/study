# 데이터 구성
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터 생성
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

# 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=1,
            validation_data= (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

# 예측
print("loss : ", loss)
print("acc : ", acc)

#loss : 11138(1) / 1.11(2)
#acc : 0.0(1) / 1.0(2)
