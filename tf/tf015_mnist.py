# 레이어 10개
# 케라스 dnn 형식으로 작성하기

import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

#1-1.그래프 수준의 랜덤 시드 설정
tf.set_random_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)
# print(y_train.shape)    # (60000,)
# print(y_test.shape)     # (10000,)

#1-2. x데이터 2차원 reshape
x_train = x_train.reshape(-1,28*28)/255
x_test = x_test.reshape(-1,28*28)/255

#1-3. y데이터 원핫인코딩
sess = tf.Session()
y_train = tf.one_hot(y_train, depth=10).eval(session=sess)
y_test = tf.one_hot(y_test, depth=10).eval(session=sess)
sess.close()
y_train = y_train.reshape(-1,10)
y_test = y_test.reshape(-1,10)

#1-2. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder('float32', [None, 28*28])
y = tf.placeholder('float32', [None, 10])

#2. 모델구성
w = tf.Variable(tf.random_normal([28*28,100]))
b = tf.Variable(tf.random_normal([100]))
layer = tf.matmul(x,w)+b

w = tf.Variable(tf.random_normal([100,50]))
b = tf.Variable(tf.random_normal([50]))
layer = tf.nn.softmax(tf.matmul(layer,w)+b)

w = tf.Variable(tf.random_normal([50,50]))
b = tf.Variable(tf.random_normal([50]))
layer = tf.nn.softmax(tf.matmul(layer,w)+b)

w = tf.Variable(tf.random_normal([50,50]))
b = tf.Variable(tf.random_normal([50]))
layer = tf.nn.softmax(tf.matmul(layer,w)+b)

w = tf.Variable(tf.random_normal([50,10]))
b = tf.Variable(tf.random_normal([10]))
h = tf.nn.softmax(tf.matmul(layer,w)+b)

#2-1. cost 손실함수(categorical_crossentropy)정의
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. cost를 최소화하는 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(301):
        _, cost_val, h_val = sess.run([opt, cost, h], feed_dict={x:x_train, y:y_train})
        if step%10==0:
            print(f"step:{step}, cost_val:{cost_val}")

    pred = tf.equal(tf.argmax(h,1),tf.argmax(y,1))
    print("pred: \n", pred, sess.run(tf.argmax(pred,1)))
    
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    print("acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test}))
