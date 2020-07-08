# mv? multi variable

import tensorflow as tf
tf.set_random_seed(777)

x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152],
          [185],
          [180],
          [205],
          [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 행렬연산이 되어야 하므로 x의 3열이 그대로 세로로 꽂혀서 3행이 됨
w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# matmul 행렬연산해주는 
h = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(h-y))
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x:x_data, y:y_data})
    if step%10==0:
        print(step, "cost: ", cost_val, "\n 예측값 : \n", h_val)
