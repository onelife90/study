import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape)    # (10, )

# RNN 모델을 짜시오

#1-1. 데이터 split
size = 6

def split_x(seq,size):
    new = []
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        new.append([item for item in subset])
    return np.array(new)

dataset = split_x(dataset, size)
print(f"dataset: \n {dataset}")     # (5,5)
# dataset:
#  [[ 1  2  3  4  5  6]
#  [ 2  3  4  5  6  7]
#  [ 3  4  5  6  7  8]
#  [ 4  5  6  7  8  9]
#  [ 5  6  7  8  9 10]]

#1-2. 데이터 슬라이싱
x_data = dataset[:, :-1]
print(f"x_data.shape:\n {x_data.shape}")    # (5,5)
y_data = dataset[:, -1] 
print(f"y_data.shape:\n {y_data.shape}")    # (5, )

#1-3. x데이터 3차원 reshape, y데이터 shape 복귀
x_data = x_data.reshape(1,5,5)
y_data = y_data.reshape(1,5)
print(f"y_data.shape: {y_data.shape}")
# y_data.shape: (1, 5)

#1-4. shape 부분을 변수명으로 저장
seq_length = 5
input_dim = 5
output = 100
# output 거의 10 이하인 숫자면 에러가 난다. 도대체 왜.. 살펴볼것
batch_size = 1

#1-5. feed_dict에 feed될 placeholder 생성
x = tf.compat.v1.placeholder(tf.float32, (None,seq_length,input_dim))
y = tf.compat.v1.placeholder(tf.int32, (None,seq_length))

#2. 모델구성
cell = tf.nn.rnn_cell.BasicLSTMCell(output)
h, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print(f"h: {h}")    # (?, 5, 100)

#3. 컴파일
w = tf.ones([batch_size, seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=h, targets=y, weights=w)
cost = tf.reduce_mean(seq_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

#4. 예측
pred = tf.argmax(h, axis=2)
print(f"pred: {pred}")  # shape=(?, 5)

#5. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(301):
        loss, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        res = sess.run(pred, feed_dict={x:x_data})
        print(f"loss: {loss}, pred: {res}, true_y: {y_data}") 
