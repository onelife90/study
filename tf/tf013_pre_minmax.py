# preprocessing

import tensorflow as tf
import numpy as np

# min_max_scaler : x-최소/최대-최소
def min_max_scaler(dataset):
    # np.min(dataset, 0) axis=0 열에서 최소값을 찾겠다
    numerator = dataset - np.min(dataset, 0)
    denominator = np.max(dataset,0)-np.min(dataset,0)
    # 어떤 수를 0으로 나누지 않는 경우를 방지
    return numerator/(denominator + 1e-7)

#1. 데이터
dataset = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

#1-1. scaler
dataset = min_max_scaler(dataset)
print(dataset)

#1-2. 데이터 슬라이싱
x_data = dataset[:, 0:-1]
y_data = dataset[:, [-1]]
# print(x_data.shape) # (8, 4)
# print(y_data.shape) # (8, 1)

#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder('float32', [None, 4])
y = tf.placeholder('float32', [None, 1])

#2. 회귀 모델 구성
w = tf.Variable(tf.random_normal([4,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))
h = tf.matmul(x,w)+b

#2-1. cost 손실함수 정의
cost = tf.reduce_mean(tf.square(h-y))

#2-2. 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

#3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, h_val, _ = sess.run([cost, h, opt], feed_dict={x:x_data, y:y_data})
        if step%10==0:
            print(step, "cost: ", cost_val, "\n 예측값: \n", h_val)
