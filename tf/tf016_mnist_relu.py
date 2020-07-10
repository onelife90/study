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
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

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
w = tf.Variable(tf.zeros([28*28,100]))
b = tf.Variable(tf.zeros([100]))
layer = tf.nn.relu(tf.matmul(x,w)+b)

w = tf.Variable(tf.zeros([100,200]))
b = tf.Variable(tf.zeros([200]))
layer = tf.nn.relu(tf.matmul(layer,w)+b)

layer = tf.nn.dropout(layer, keep_prob=0.2)

w = tf.Variable(tf.zeros([200,50]))
b = tf.Variable(tf.zeros([50]))
layer = tf.nn.relu(tf.matmul(layer,w)+b)

w = tf.Variable(tf.random_normal([50,50]))
b = tf.Variable(tf.random_normal([50]))
layer = tf.nn.relu(tf.matmul(layer,w)+b)

w = tf.Variable(tf.random_normal([50,10]))
b = tf.Variable(tf.random_normal([10]))
h = tf.nn.softmax(tf.matmul(layer,w)+b)

#2-1. cost 손실함수(categorical_crossentropy)정의
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. cost를 최소화하는 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=0.035).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(301):
        _, cost_val, h_val = sess.run([opt, cost, h], feed_dict={x:x_train, y:y_train})

        if step%30==0:
            print(f"step:{step}, cost_val:{cost_val}")
            # step:0, cost_val:8.272578239440918
            # step:30, cost_val:3.176189661026001
            # step:60, cost_val:2.3015921115875244
            # step:90, cost_val:2.301161050796509
            # step:120, cost_val:2.30116605758667
            # step:150, cost_val:2.30116605758667
            # step:180, cost_val:2.30116605758667
            # step:210, cost_val:2.30116605758667
            # step:240, cost_val:2.30116605758667
            # step:270, cost_val:2.30116605758667
            # step:300, cost_val:2.30116605758667

    # tf.argmax(h,1)==예측값의 1(행)을 기준으로 최대값과 tf.argmax(y,1)==실제값의 1(행)을 기준으로 최대값이 같은 것을 pred로 정의
    pred = tf.equal(tf.argmax(h,1),tf.argmax(y,1))
    
    # 예측값을 실수형으로 캐스팅하여 차원을 모두 제거하고 평균을 낸 acc 정의
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    
    print("acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test}))
    # acc:  0.1135
