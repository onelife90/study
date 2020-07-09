# 다중분류

import tensorflow as tf
import numpy as np

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

#1-1. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder('float32', [None, 4])
y = tf.placeholder('float32', [None, 3])

#2. 모델 구성
w = tf.Variable(tf.random_normal([4,3]), name='weight')
# y 컬럼이 3개이기 때문에 shape를 3으로 맞춰줘야함
b = tf.Variable(tf.random_normal([1,3]), name='bias')

# keras110_9_softmax.py 원그래프 참조. 합쳐서 1이 나오게 변경
h = tf.nn.softmax(tf.matmul(x,w)+b)

#2-1. cost 손실함수(categorical_crossentropy) 정의
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. loss를 최소화하는 옵티마이저 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([opt, loss], feed_dict={x:x_data, y:y_data})
    
        if step%200==0:
            print(step, "cost_val: " , cost_val)

    # 최적의 w와 b가 구해져있다
    # 새로운 x 데이터를 feed하여 예측해보자
    a = sess.run(h, feed_dict={x:[[1,11,7,9]]})
    print("a의 예측값", a, "a의 예측값 중 최대값: ", sess.run(tf.argmax(a,1)))
    # [[8.8999486e-01 1.0947233e-01 5.3285377e-04]] [0] 
    
    b = sess.run(h, feed_dict={x:[[1,3,4,3]]})
    print("b의 예측값", b, "b의 예측값 중 최대값: ", sess.run(tf.argmax(b,1)))
    # b의 예측값 [[0.19012941 0.61809987 0.1917707 ]] b의 예측값 중 최대값:  [1]

    c = sess.run(h, feed_dict={x:[[11,33,4,3]]})
    print("c의 예측값", c, "c의 예측값 중 최대값: ", sess.run(tf.argmax(c,1)))
    # c의 예측값 [[4.5390652e-13 5.6907021e-09 1.0000000e+00]] c의 예측값 중 최대값:  [2]

    # a,b,c를 넣어서 완성할 것 / 힌트:feed_dict를 수정하세요
    all = sess.run(h, feed_dict={x:[[1,11,7,9],
                                    [1,3,4,3],
                                    [3,5,7,11],
                                    [11,33,4,13]]})
    print("all의 예측값", all, "all의 예측값 중 최대값: ", sess.run(tf.argmax(all,1)))
#  [1.1053275e-01 6.5884113e-01 2.3062603e-01]      
#  [3.5355780e-02 9.6450239e-01 1.4193769e-04]      
#  [1.8095319e-13 1.4422866e-08 1.0000000e+00]] all의 예측값 중 최대값:  [0 1 1 2]
