# mv? multi variable

import tensorflow as tf
tf.set_random_seed(777) # 잭 팟

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# x가 3개이므로 각각의 weight가 존재
w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# column이 여러 개인 것과 같으니까 각각의 가중치를 곱해서 마지막에 bias를 더해주면 된다
h = (x1 * w1) + (x2 * w2) + (x3 * w3) + b

# 손실과 옵티마이저 설정 
cost = tf.reduce_mean(tf.square(h-y))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(tf.global_variables_initializer())
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if step%10==0:
        print(step, "cost: ", cost_val, "\n", h_val)
