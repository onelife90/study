# 다중분류
# iris 코드를 완성하시오

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
iris = load_iris()
x_data = iris.data
y_data = iris.target
# print(x_data.shape)     # (150, 4)
# print(y_data.shape)     # (150, )
# print(y_data)           # 0,1,2 3개분류

#1-1. y데이터 원핫인코딩
sess = tf.Session()
y_data = tf.one_hot(y_data, depth=3).eval(session=sess)
# print(y_data.shape)     # (150, 3)
y_data = y_data.reshape(-1,3)

#1-2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=88, train_size=0.8)
# print("x_train, x_test", x_)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

#2. 모델 구성
w = tf.Variable(tf.random_normal([4,3]), name='weight')
# y 컬럼이 3개이기 때문에 shape를 3으로 맞춰줘야함
b = tf.Variable(tf.random_normal([3]), name='bias')

# keras110_9_softmax.py 원그래프 참조. 합쳐서 1이 나오게 변경
h = tf.nn.softmax(tf.matmul(x,w)+b)
# print("h: ", h)
# h:  Tensor("Softmax:0", shape=(?, 3), dtype=float32)      

#2-1. cost 손실함수(categorical_crossentropy) 정의
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. loss를 최소화하는 옵티마이저 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=2e-2).minimize(loss)

#3. 훈련
# 각 session에 컨텍스트 매니저가 있어서 with 구문 끝에서 자동으로 close()가 호출
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, h_val, cost_val = sess.run([opt, h, loss], feed_dict={x:x_train, y:y_train})

        if step%200==0:
            print(step, "cost_val: " , cost_val)
        #    0 cost_val:  6.311684
        #     200 cost_val:  0.4222794
        #     400 cost_val:  0.34365538
        #     600 cost_val:  0.3002386
        #     800 cost_val:  0.26958433
        #     1000 cost_val:  0.24620421
        #     1200 cost_val:  0.2276606
        #     1400 cost_val:  0.21256508
        #     1600 cost_val:  0.20003031
        #     1800 cost_val:  0.18945326
        #     2000 cost_val:  0.18040702
    
    # tf.argmax(h,1)==h의 1(행)을 기준으로 최대값과 tf.argmax(y,1)==y의 1(행)을 기준으로 최대값이 같은 것을 pred로 지정
    pred = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
    
    # pred와 y를 실수형으로 캐스팅해서 차원을 제거한 후 평균으로 acc 구하기
    acc = tf.reduce_mean(tf.cast(pred, dtype=tf.float32))
    print("Acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test}))
    # Acc:  0.93333334
