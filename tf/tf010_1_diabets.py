# 회귀

from sklearn.datasets import load_diabetes
import tensorflow as tf

#1. 데이터
diabets = load_diabetes()
x_data = diabets.data
y_data = diabets.target
print(x_data.shape) # (442, 10)
print(y_data.shape) # (442, )

y_data = y_data.reshape(-1,1)
print(y_data.shape) # (442, 1)

#1-1. feed_dict에 feed  될 텐서를 위한 placeholder 설정
x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

#2. 모델 구성
w = tf.Variable(tf.random_normal([10,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))
h = tf.matmul(x,w)+b

#2-1. cost 손실함수 정의
cost = tf.reduce_mean(tf.square(h-y))

#2-2. 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=4e-2)
train = opt.minimize(cost)

#3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x:x_data, y:y_data})
        if step%100==0:
            print(step, "cost: ", cost_val, "\n 예측값: \n", h_val)
