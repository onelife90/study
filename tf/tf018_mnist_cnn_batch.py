'''제일 작은 괄호부터 생각해서 벗어나기
import numpy as np
a = np.array([[[1]], [[1]]])
print(a.shape)  # (2, 1, 1)

a = np.array([[[[1]], [[1]]],
              [[[1]], [[1]]]])
print(a.shape)  # (2, 2, 1, 1)
'''

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

#1-2. x데이터 2차원 reshape + 정규화
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255

#1-3. y데이터 원핫인코딩
sess = tf.Session()
y_train = tf.one_hot(y_train, depth=10).eval(session=sess)
y_test = tf.one_hot(y_test, depth=10).eval(session=sess)
sess.close()
y_train = y_train.reshape(-1,10)
y_test = y_test.reshape(-1,10)

#1-4. 튜닝할 부분 변수명 저장
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100

#1-5. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder('float32', [None, 28,28,1])
# x = tf.reshape([x, [-1,28,28,1])
y = tf.placeholder('float32', [None, 10])
# keep_prob = tf.placeholder('float32')   # dropout

#2. 모델구성
# shape=[3,3,1,32] kernel_size(3,3) channel=1, opyput=32
w = tf.get_variable("w1", shape=[3,3,1,32])#, initializer=tf.contrib.layers.xavier_initializer())
# 케라스 버전 : Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1))
# print("w: ", w)
# w:  <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>

# Conv2D에는 bias 계산이 자동으로 되기 때문에 b 따로 명시 안해줘도 됨
# b = tf.Variable(tf.random_normal([512]))

# strides=[1,1,1,1] 양 옆에 1은 기본. 결국 strides=[1,1]라는 의미
layer = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
# print("layer: ", layer)
# layer:  Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

layer = tf.nn.selu(layer)
# print("layer: ", layer)
# layer:  Tensor("Selu:0", shape=(?, 28, 28, 32), dtype=float32)

layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# print("layer: ", layer)
# layer:  Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)


w = tf.get_variable("w2", shape=[3,3,32,64])#, initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([512]))
layer = tf.nn.conv2d(layer, w, strides=[1,1,1,1], padding='SAME')
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# print("layer: ", layer)
# layer:  Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

layer = tf.reshape(layer, [-1,7*7*64])

w = tf.get_variable("w3", shape=[7*7*64, 10])#, initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
h = tf.nn.softmax(tf.matmul(layer,w)+b)

'''
layer = tf.nn.dropout(layer, keep_prob=keep_prob)
print("layer: ", layer)

w = tf.get_variable("w2", shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([512]))
layer = tf.nn.selu(tf.matmul(layer,w)+b)
layer = tf.nn.dropout(layer, keep_prob=keep_prob)

w = tf.get_variable("w3", shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([512]))
layer = tf.nn.selu(tf.matmul(layer,w)+b)
layer = tf.nn.dropout(layer, keep_prob=keep_prob)

w = tf.get_variable("w4", shape=[512,256], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([256]))
layer = tf.nn.selu(tf.matmul(layer,w)+b)
layer = tf.nn.dropout(layer, keep_prob=keep_prob)

w = tf.get_variable("w5", shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
h = tf.nn.softmax(tf.matmul(layer,w)+b)
'''
#2-1. cost 손실함수(categorical_crossentropy)정의
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. cost를 최소화하는 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs): # 15
        avg_cost = 0

        for i in range(total_batch):    # 600
            start = i * batch_size
            end = start + batch_size
            # 배치사이즈 완성하시오 0~99/100~199/200~299/.../59900~59999 슬라이싱 사용
            
            # batch_xs, batch_ys = x_train[:100], y_train[:100]
            # batch_xs, batch_ys = x_train[100:200], y_train[100:200]
            
            # batch_xs, batch_ys = x_train[i:batch_size], y_train[i:batch_size]
            # batch_xs, batch_ys = x_train[i*batch_size:batch_size*(i+1)], y_train[i*batch_size:batch_size(i+2)]
            
            batch_xs, batch_ys = x_train[start:end], y_train[start:end]

            feed_dict = {x:batch_xs, y:batch_ys}
            c, _ = sess.run([cost, opt], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print(f"epoch: {epoch+1}\t cost: {avg_cost}")
        # epoch: 1         cost: 2.2671798976262396
        # epoch: 2         cost: 1.7514511213699968
        # epoch: 3         cost: 1.0651708681384722
        # epoch: 4         cost: 0.6751592071106033
        # epoch: 5         cost: 0.5232279736797013
        # epoch: 6         cost: 0.4487951478362082
        # epoch: 7         cost: 0.40442931114385533
        # epoch: 8         cost: 0.3743016773710647
        # epoch: 9         cost: 0.35195455490301053
        # epoch: 10        cost: 0.33433568277085784
        # epoch: 11        cost: 0.31982452747722473
        # epoch: 12        cost: 0.30747287606820434
        # epoch: 13        cost: 0.2966904300513369
        # epoch: 14        cost: 0.2870930494430164
        # epoch: 15        cost: 0.27840917440131296
    print("훈련 끝!")

    # tf.argmax(h,1)==예측값의 1(행)을 기준으로 최대값과 tf.argmax(y,1)==실제값의 1(행)을 기준으로 최대값이 같은 것을 pred로 정의
    pred = tf.equal(tf.arg_max(h,1), tf.argmax(y,1))

    # 예측값을 실수형으로 캐스팅하여 차원을 모두 제거하고 평균을 낸 acc 정의
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    print("acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test}))
    # acc:  0.9271
