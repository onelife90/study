import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [3.,5.,7.]

w = tf.placeholder(tf.float32)
h = x * w
cost = tf.reduce_mean(tf.square(h-y))

w_hist = []
cost_hist = []

with tf.Session() as sess:
    for i in range(-30, 50):
        # plt의 가로축 간격
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_hist.append(curr_w)
        cost_hist.append(curr_cost)

plt.plot(w_hist, cost_hist)
plt.grid()
plt.show()
