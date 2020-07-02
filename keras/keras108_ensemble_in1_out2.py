# 여태 한 모델은 분류와 회귀 나눠서 실행
# 만약 분류와 회귀를 동시에 넣어서 실행한다면?
#ex)
# x == 우리 반 학생들의 키
# y1 == 키, y2 == 성별

# 결론은 잘 안나옴

#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(1, ))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

x2 = Dense(50)(x1)
output1 = Dense(1)(x2)
# Dense_5

x3 = Dense(70)(x1)
x3 = Dense(70)(x3)
output2 = Dense(1, activation='sigmoid')(x3)
# Dense_8

model = Model(inputs=input1, outputs=[output1,output2])

# model.summary()

#3. 컴파일, 훈련
model.compile(loss=['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse','acc'])
model.fit(x_train,[y1_train,y2_train], epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_train, [y1_train, y2_train])
print("loss: ", loss)
# loss:  [0.6822894811630249, 0.0013018224854022264, 0.6809876561164856, 0.0013018224854022264, 1.0, 0.24397888779640198, 0.6000000238418579]
# loss 총 7개 순서대로
# 전체 모델의 loss == dense_5_loss + dense_8_loss
# dense_5_loss
# dense_8_loss
# dense_5_mse
# dense_5_acc
# dense_8_mse
# dense_8_acc

x1_pred = np.array([11,12,13,14])
y_pred = model.predict(x1_pred)
print("y_pred: \n", y_pred)
# [array([[11.015211],
#        [12.015242],
#        [13.015271],
#        [14.015303]], dtype=float32)

# array([[0.36985648],        
#        [0.35025704],
#        [0.33115035],
#        [0.31258458]], dtype=float32)]
