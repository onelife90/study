# h에 활성화 함수 사용해보자

import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 행렬연산이 되어야 하므로 x의 3열이 그대로 세로로 꽂혀서 3행이 됨
w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 행렬 연산해주는 matmul을 랩핑해서 sigmoid 활성화 함수 사용
# sigmoid는 엄밀히 말하면 linear regressor
h = tf.sigmoid(tf.matmul(x, w) + b)

# binary_crossentropy를 표현한 식
cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)

# h>0.5이면 예측을 하겠다
pred = tf.cast(h>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step%10==0:
            print(step, "cost: ", cost_val)

    h, c, a = sess.run([h, pred, acc], feed_dict={x:x_data, y:y_data})
    print("\n H : \n", h, "\n correct(y) : \n", c ,"ACC : ", a)
