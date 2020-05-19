# keras15_mlp2.py를 Sequential에서 함수형으로 변경
# earlyStopping 적용

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array(range(711,811))

x = np.transpose(x)
y = np.transpose(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=99, train_size=0.6
)      

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))
dense1 = Dense(100)(input1)
dense1 = Dense(10)(input1)
dense1 = Dense(9000)(input1)
dense1 = Dense(10)(input1)

output1 = Dense(100)(dense1)
output1 = Dense(80)(output1)
output1 = Dense(60)(output1)
output1 = Dense(40)(output1)
output1 = Dense(20)(output1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)

model = Model(inputs=input1, outputs=output1)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x_train, y_train, epochs=500, batch_size=1,
            validation_split=0.3, verbose=3, 
            callbacks=[early_stopping])
           
#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
       
print("RMSE : ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print("R2 : ", r2)

# 하이퍼파라미터튜닝
