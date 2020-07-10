# 과제 cifar10 cnn 모델

import tensorflow as tf
from keras.datasets import cifar10

#1-1.그래프 수준의 랜덤 시드 설정
tf.set_random_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(f"x_train : {x_train.shape}, x_test:{x_test.shape}") 
# x_train : (50000, 32, 32, 3), x_test:(10000, 32, 32, 3)
# print(f"y_train : {y_train.shape}, y_test:{y_test.shape}")
# y_train : (50000, 1), y_test:(10000, 1)

#1-2. x데이터 2차원 정규화
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#1-3. y데이터 원핫인코딩
sess = tf.compat.v1.Session()
y_train = tf.one_hot(y_train, depth=10).eval(session=sess)
y_test = tf.one_hot(y_test, depth=10).eval(session=sess)
sess.close()
y_train = y_train.reshape(-1,10)
y_test = y_test.reshape(-1,10)

#1-4. 튜닝할 부분 변수명 저장
learning_rate = 0.001
training_epochs = 20
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 50000/100
ksize = 2
stride = 2
node = 32

#1-5. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.compat.v1.placeholder('float32', [None, 32,32,3])
# x_dense = tf.reshape(x, [-1,x.shape[1]*x.shape[2]*x.shape[3]])
y = tf.compat.v1.placeholder('float32', [None, 10])

#2. 모델구성
w = tf.compat.v1.get_variable("w1", shape=[ksize,ksize,3,node*2])
layer = tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding='SAME')
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool2d(layer, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding='SAME')

w = tf.compat.v1.get_variable("w2", shape=[ksize+1,ksize+1,layer.shape[3],node*3])
layer = tf.nn.conv2d(layer, w, strides=[1,stride,stride,1], padding='SAME')
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool2d(layer, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding='SAME')
# print(f"layer: {layer}")
# layer:  Tensor("MaxPool_1:0", shape=(?, 3, 3, 96), dtype=float32)

w = tf.compat.v1.get_variable("w3", shape=[ksize,ksize,layer.shape[3],node*4])
layer = tf.nn.conv2d(layer, w, strides=[1,stride,stride,1], padding='SAME')
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool2d(layer, ksize=[1,ksize,ksize,1], strides=[1,stride+1,stride+1,1], padding='SAME')

layer = tf.reshape(layer, [-1,layer.shape[1]*layer.shape[2]*layer.shape[3]])

w = tf.compat.v1.get_variable("w4", shape=[layer.shape[1], node])
layer = tf.nn.selu(tf.matmul(layer, w))

w = tf.compat.v1.get_variable("w5", shape=[node,10])
b = tf.Variable(tf.random_normal([10]))
h = tf.nn.softmax(tf.matmul(layer,w)+b)

#2-1. cost 손실함수(categorical_crossentropy)정의
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. cost를 최소화하는 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs): # 20
        avg_cost = 0

        for i in range(total_batch):    # 500
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

    print("훈련 끝!")

    # tf.argmax(h,1)==예측값의 1(행)을 기준으로 최대값과 tf.argmax(y,1)==실제값의 1(행)을 기준으로 최대값이 같은 것을 pred로 정의
    pred = tf.equal(tf.arg_max(h,1), tf.argmax(y,1))

    # 예측값을 실수형으로 캐스팅하여 차원을 모두 제거하고 평균을 낸 acc 정의
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    print("acc: ", sess.run(acc, feed_dict={x:x_test, y:y_test}))