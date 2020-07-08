# mv? multi variable
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

dataset = np.loadtxt('./data/csv/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = dataset[:, 0:-1]
y_data = dataset[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 행렬연산이 되어야 하므로 x의 3열이 그대로 가로로 꽂혀서 3행이 됨
w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# matmul 행렬연산해주는 함수
# 활성화 함수 linear가 h에 포함
h = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(h-y))
opt = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x:x_data, y:y_data})
    if step%10==0:
        print(step, "cost: ", cost_val, "\n 예측값 : \n", h_val)
