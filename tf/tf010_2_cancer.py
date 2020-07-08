# 이진분류

from sklearn.datasets import load_breast_cancer
import tensorflow as tf

#1. 데이터
cancer = load_breast_cancer()
x_data = cancer.data
y_data = cancer.target
# print(x_data.shape) # (569, 30)
# print(y_data.shape) # (569,)

y_data = y_data.reshape(-1,1)
# print(y_data.shape) # (569,1)

#1-1. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

#2. 모델 구성
w = tf.Variable(tf.random_normal([30,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))
h = tf.matmul(x,w) + b

#2-1. cost 손실함수(binary_crossentropy) 정의
cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

opt = tf.train.GradientDescentOptimizer(learning_rate=8e-6)
train = opt.minimize(cost)

#2-2. 예측
# h>0.5면 예측을 하겠다
pred = tf.cast(h>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

#3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
    h,c,a = sess.run([h, pred, acc], feed_dict={x:x_data, y:y_data})
    print("H: \n", h, "\n correct(y): \n", c, "\n acc: ", a)
    #  acc:  0.8383128
