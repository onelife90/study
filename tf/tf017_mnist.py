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
b = tf.Variable(tf.random_normal([512]))
layer = tf.nn.selu(tf.matmul(x,w)+b)
layer = tf.nn.dropout(layer, keep_prob=keep_prob)

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

#2-1. cost 손실함수(categorical_crossentropy)정의
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))

#2-2. cost를 최소화하는 최적화 함수 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_epochs):
        avg_cost = 0

        for i in range(total_batch):
            # 배치사이즈 완성하시오
            batch_xs, batch_ys = x_train([batch_size])
            
            feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.7}
            c, _ = sess.run([cost, opt], feed_dict=feed_dict)
            avg_cost += c /total_batch
        print('epoch: ', '%04d' %(epoch+1), 'cost: ', {".9f"}.fomat(avg_cost))
print("훈련 끝!")

pred = tf.equal(tf.arg_max(h,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(pred, tf.float32))
print("acc: ", acc)
