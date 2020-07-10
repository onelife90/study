# 레이어 10개
# 케라스 dnn 형식으로 작성하기
# batch_size 줘보자

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
x_train = x_train.reshape(-1,28*28).astype('float32')/255
x_test = x_test.reshape(-1,28*28).astype('float32')/255

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
x = tf.placeholder('float32', [None, 28*28])
y = tf.placeholder('float32', [None, 10])
keep_prob = tf.placeholder('float32')   # dropout

#2. 모델구성
w = tf.get_variable("w1", shape=[784,512], initializer=tf.contrib.layers.xavier_initializer())
# print("w: " , w)
# w <tf.Variable 'w1:0' shape=(784, 512) dtype=float32_ref>

b = tf.Variable(tf.random_normal([512]))
# print("b: ", b)
# b:  <tf.Variable 'Variable:0' shape=(512,) dtype=float32_ref>

layer = tf.nn.selu(tf.matmul(x,w)+b)
# print("selu.layer: ", layer)
# selu.layer:  Tensor("Selu:0", shape=(?, 512), dtype=float32)

layer = tf.nn.dropout(layer, keep_prob=keep_prob)
# print("dropout_layer: ", layer)
# dropout_layer:  Tensor("dropout/mul_1:0", shape=(?, 512), dtype=float32)

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
# print("h: ", h)
# h:  Tensor("Softmax:0", shape=(?, 10), dtype=float32)     

#2-1. cost 손실함수(categorical_crossentropy)정의
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. cost를 최소화하는 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#3. 훈련
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

            feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.7}
            c, _ = sess.run([cost, opt], feed_dict=feed_dict)
            avg_cost += c /total_batch

        print('epoch: ', '%04d' %(epoch+1), "cost: {:.9f}".format(avg_cost))
        # epoch:  0001 cost: 3.179819596
        # epoch:  0002 cost: 1.863974968
        # epoch:  0003 cost: 1.338062303
        # epoch:  0004 cost: 1.086432286
        # epoch:  0005 cost: 0.954928038
        # epoch:  0006 cost: 0.858678305
        # epoch:  0007 cost: 0.788396372
        # epoch:  0008 cost: 0.737423236
        # epoch:  0009 cost: 0.701132693
        # epoch:  0010 cost: 0.668992661
        # epoch:  0011 cost: 0.638948249
        # epoch:  0012 cost: 0.618771437
        # epoch:  0013 cost: 0.604265884
        # epoch:  0014 cost: 0.580154929
        # epoch:  0015 cost: 0.565156868
        print("훈련 끝!")

    # tf.argmax(h,1)==예측값의 1(행)을 기준으로 최대값과 tf.argmax(y,1)==실제값의 1(행)을 기준으로 최대값이 같은 것을 pred로 정의
    pred = tf.equal(tf.argmax(h,1),tf.argmax(y,1))

    # 예측값을 실수형으로 캐스팅하여 차원을 모두 제거하고 평균을 낸 acc 정의
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    print("acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test, keep_prob:0.9}))
    # acc:  0.8933
