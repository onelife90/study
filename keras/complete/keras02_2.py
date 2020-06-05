# 데이터 구성
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터 생성
x_train = np.array([1,3,5,7,9,11,13,15,17,19])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
x_test = np.array([111,113,115,117,119,121,123,125,127,129])
y_test = np.array([200,202,204,206,208,210,212,214,216,218])

# 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(2))
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

output = model.predict(x_test)
print("결과물 : \n", output)

'''
#epochs=100, batch_size=1, Dense=5,3,2,1
#결과물
 [[112.74669 ]
 [114.761894]
 [116.77709 ]
 [118.7923  ]
 [120.80749 ]
 [122.82269 ]
 [124.8379  ]
 [126.8531  ]
 [128.8683  ]
 [130.8835  ]]
 
#epochs=200, batch_size=1, Dense=5,3,2,1
#결과물
 [[111.93523 ]
 [113.9339  ]
 [115.93257 ]
 [117.931244]
 [119.92991 ]
 [121.92858 ]
 [123.92726 ]
 [125.92594 ]
 [127.924614]
 [129.9233  ]]
 
#epochs=200, batch_size=1, Dense=10,8,6,4,2,1
#결과물
 [[112.18175]
 [114.18543]
 [116.18915]
 [118.19285]
 [120.19655]
 [122.20025]
 [124.20393]
 [126.20765]
 [128.21135]
 [130.21504]] 

#epochs=100, batch_size=1, Dense=20,18,16,...,4,2,1
#결과물
 [[112.089905]
 [114.09166 ]
 [116.09345 ]
 [118.09524 ]
 [120.097   ]
 [122.098755]
 [124.10053 ]
 [126.102295]
 [128.1041  ]
 [130.10587 ]]
 loss : 7726
 acc : 0.0
 
 #epochs=50, batch_size=1, Dense=5,9,7,50,3,2,1
loss :  7832.87314453125
acc :  0.0
결과물 :
 [[111.53878 ]
 [113.529396]
 [115.52    ]
 [117.5106  ]
 [119.50118 ]
 [121.49178 ]
 [123.48237 ]
 [125.47297 ]
 [127.463585]
 [129.45416 ]]
'''
