# x,y,w,b, h,cost,opt
# sigmoid 사용
# pred, acc 준비

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
w = tf.Variable(tf.random_normal([2,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))
h = tf.matmul(x,w) + b

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
    # H:
    # [[0.27415314]
    # [2.247741  ]
    # [2.3320005 ]
    # [4.3055882 ]]
    # correct(y):
    # [[0.]
    # [1.]
    # [1.]
    # [1.]]
    # acc:  0.75
