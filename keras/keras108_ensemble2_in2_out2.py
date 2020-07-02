# 여태 한 모델은 분류와 회귀 나눠서 실행
# 만약 분류와 회귀를 동시에 넣어서 실행한다면?
#ex)
# x == 우리 반 학생들의 키
# y1 == 키, y2 == 성별

# 그렇다면 2개의 인풋과 2개의 아웃풋으로하면 명확하게 나오지 않을까?
# 실습해보자
# 결론은 잘 안나옴

#1. 데이터
import numpy as np
x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1, ))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(1, ))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1,x2])

x3 = Dense(100)(merge)
output1 = Dense(1)(x3)

x4 = Dense(70)(merge)
x4 = Dense(70)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs=[input1,input2], outputs=[output1,output2])

# model.summary()

#3. 컴파일, 훈련
model.compile(loss=['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse','acc'])
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate([x1_train,x2_train], [y1_train, y2_train])
print("loss: ", loss)
# loss:  [0.6800873875617981, 9.104357013711706e-05, 0.6799963712692261, 9.104357013711706e-05, 1.0, 0.24352975189685822, 0.5]

# loss 총 7개 순서대로
# 전체 모델의 loss == dense_8_loss: + dense_11_loss:
# dense_8_loss
# dense_11_loss
# dense_8_mse:
# dense_8_acc:
# dense_11_mse:
# dense_11_acc:

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred,x2_pred])
print("y_pred: \n", y_pred)

# [array([[10.98703 ],
#        [11.98637 ],
#        [12.985713],
#        [13.985055]], dtype=float32)

# array([[0.3661247 ],        
#        [0.34646556],
#        [0.32731688],
#        [0.3087265 ]], dtype=float32)]
