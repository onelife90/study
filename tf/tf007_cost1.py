import tensorflow as tf
import matplotlib.pyplot as plt

#1. 데이터
x = [1.,2.,3.]
y = [3.,5.,7.]

#1-2. feed_dict에 feed 될 텐서를 위한 placeholder 설정
w = tf.placeholder(tf.float32)
h = x * w

#2. cost 손실 함수 정의
cost = tf.reduce_mean(tf.square(h-y))

#2-1. 갱신되는 값 추정
w_hist = []
cost_hist = []

with tf.Session() as sess:
    for i in range(-30, 50):
        # plt의 가로축 간격(-3~5)
        curr_w = i * 0.1
        # 세로축은 feed_dict에 (-3~5)를 넣어서 cost 계산
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_hist.append(curr_w)
        cost_hist.append(curr_cost)

plt.plot(w_hist, cost_hist)
plt.grid()
plt.show()
