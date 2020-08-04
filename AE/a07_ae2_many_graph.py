# 재사용 가능한 함수로 모델을 여러개 만들어보자

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

#1. 데이터
train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

#1-1. 데이터 전처리(정규화)
x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2])/255

#2. 모델 구성
# 함수로 autoencoder 모델 구성
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model1 = autoencoder(hidden_layer_size=1)
model2 = autoencoder(hidden_layer_size=2)
model4 = autoencoder(hidden_layer_size=4)
model8 = autoencoder(hidden_layer_size=8)
model16 = autoencoder(hidden_layer_size=16)
model32 = autoencoder(hidden_layer_size=32)

#3. 컴파일, 훈련
model1.compile(loss='mse', optimizer='adam', metrics=['acc'])   
model1.fit(x_train, x_train, epochs=50)

model2.compile(loss='mse', optimizer='adam', metrics=['acc'])   
model2.fit(x_train, x_train, epochs=50)

model4.compile(loss='mse', optimizer='adam', metrics=['acc'])   
model4.fit(x_train, x_train, epochs=50)

model8.compile(loss='mse', optimizer='adam', metrics=['acc'])  
model8.fit(x_train, x_train, epochs=50)

model16.compile(loss='mse', optimizer='adam', metrics=['acc'])   
model16.fit(x_train, x_train, epochs=50)

model32.compile(loss='mse', optimizer='adam', metrics=['acc'])   
model32.fit(x_train, x_train, epochs=50)

#4. 예측
output1 = model1.predict(x_test)
output2 = model2.predict(x_test)
output4 = model4.predict(x_test)
output8 = model8.predict(x_test)
output16 = model16.predict(x_test)
output32 = model32.predict(x_test)

#5. 시각화 - 한번에 여러그래프 그리기
# matplotlib로 그래프를 그리려면 Figure 객체와 하나 이상의 subplot(Axes) 객체가 필요
fig, axes =  plt.subplots(7,5,figsize=(15,15))

print(f'type(axes): {type(axes)}')  # type(axes): <class 'numpy.ndarray'>
print(f'axes.shape: {axes.shape}')  # axes.shape: (7, 5)   


# 10000개 중에서 5개를 랜덤으로 뽑을것이야
random_img = random.sample(range(output1.shape[0]), 5)
outputs = [x_test,output1,output2,output4,output8,output16,output32]

# enumerate 반복문 사용시 몇 번째 반복문인지 확인
for row_num, row in enumerate(axes):
    # print(f'row_num: {row_num}')        # row_num: 0~6 총 7개
    # print(f'row.shape: {row.shape}')    # row.shape: (5,)
    for col_num, ax in enumerate(row):
        # print(f'col_num: {col_num}')    # col_num: 0~4 총 5개
        # print(f'ax: {ax}')              # ax: AxesSubplot(0.285345,0.335366;0.133621x0.0939024) 좌표라 생각하면 쉬울듯    
        
        # row_num==0(x_test) & random_img[0], row_num==0 & random_img[1],...,row_num==0 & random_img[4]
        # row_num==1(output1) & random_img[0], row_num==0 & random_img[1],...,row_num==0 & random_img[4]
        # ... 이렇게 output32까지 진행됨
        ### 즉, 7행 5열의 그림을 볼 수 있음
        ax.imshow(outputs[row_num][random_img[col_num]].reshape(28,28), cmap='gray')
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
