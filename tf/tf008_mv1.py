# mv? multi variable

import tensorflow as tf

#1-1. 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

#1-2. 데이터
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#2. 모델 구성
# x가 3개이므로 각각의 weight가 존재
w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# column이 여러 개인 것과 같으니까 각각의 가중치를 곱해서 마지막에 bias를 더해주면 된다
h = (x1 * w1) + (x2 * w2) + (x3 * w3) + b

#2-1. cost 손실 함수 정의
cost = tf.reduce_mean(tf.square(h-y))

#2-2. cost를 최소화하는 옵티마이저 정의. minimize(cost)=cost가 가장 적을 때 구하라
opt = tf.train.GradientDescentOptimizer(learning_rate=0.04)
train = opt.minimize(cost)

#3. 훈련(with문 X)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(tf.global_variables_initializer())
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if step%10==0:
        print(step, "cost: ", cost_val, "\n h_val", h_val)
