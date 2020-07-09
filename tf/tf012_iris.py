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
sess.close()
print(y_data.shape)     # (150, 3)

#1-2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=88, train_size=0.8)

#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
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
opt = tf.train.GradientDescentOptimizer(learning_rate=2e-2).minimize(loss)

#3. 평가, 예측
pred = tf.cast(tf.math.argmax(h), dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred,y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _, cost_val, _ = sess.run([opt, loss, acc], feed_dict={x:x_train, y:y_train})
        if step%200==0:
            print(step, "cost_val: \n" , cost_val, "\n pred: \n", acc)
