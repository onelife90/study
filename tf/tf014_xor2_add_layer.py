# 레이어를 추가해보자

import tensorflow as tf
import numpy as np

#1-1. 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

#1-2. 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder('float32', [None, 2])
y = tf.placeholder('float32', [None, 1])

#2. 모델 구성
w1 = tf.Variable(tf.random_normal([2,100], name='weight1'))
# 케라스 버전 : model.add(Dense(100, input_dim=2))

b1 = tf.Variable(tf.random_normal([100], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)
# 레이어마다 x, b 그리고 활성화 함수가 있으니 추가만 하면 됨

w2 = tf.Variable(tf.random_normal([100,50], name='weight2'))
# model.add(Dense(50))

b2 = tf.Variable(tf.random_normal([50], name='bias2'))
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2)

w3 = tf.Variable(tf.random_normal([50,1], name='weight3'))
# model.add(Dense(1))
b3 = tf.Variable(tf.random_normal([1], name='bias3'))
h = tf.sigmoid(tf.matmul(layer2,w3) + b3)


#2-1. cost 손실함수(binary_crossentropy) 정의
cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

#2-2. loss를 최소화하는 옵티마이저 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

#2-3. 예측
# 예측값이 0.5 이상인 것들만 실수형으로 캐스팅하여 pred로 정의
pred = tf.cast(h>0.5, dtype=tf.float32)
# tf.equal(pred, y)=예측값과 실제값이 같은 놈들을 찾아 실수형으로 캐스팅하고 차원을 모두 제거하여 평균을 낸 acc 정의
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

#3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, opt], feed_dict={x:x_data, y:y_data})

    h,c,a = sess.run([h, pred, acc], feed_dict={x:x_data, y:y_data})
    print("H: \n", h, "\n correct(y): \n", c, "\n acc: ", a)
    #  H:
    #  [[0.35366654]
    #  [0.71954256]
    #  [0.75838965]
    #  [0.20254518]]
    #  correct(y):
    #  [[0.]
    #  [1.]
    #  [1.]
    #  [0.]]
    #  acc:  1.0
