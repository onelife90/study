# x=hihello / y=ihello 를 예측하는 rnn 모델 구성(자기회귀)
# 레거시한 방법은 if문으로 구성을 할수도 있지만 데이터(문자)를 수치화하고 shape를 맞춰서 모델 구성

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#1. 데이터 hihello 한글자씩 인덱스 넣어주기 위해 구분
idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1,1)
print(f"_data.shape : {_data.shape}")   # _data.shape : (7, 1)
print(f"_data: \n{_data}")     
# _data:
# [['h']
#  ['i']
#  ['h']
#  ['e']
#  ['l']
#  ['l']
#  ['o']]
print(f"type_data: {type(_data)}")      # type_data: <class 'numpy.ndarray'>   

#1-2. 원핫인코딩, reshape
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
# enc.fit_transform(_data).toarray()
# 여기서, fit_transform을 사용하면 아예 먹히지 않는데 fit에 대한 입력이 int 유형의 입력 배열이기 때문이다

print(f"enc_data : \n{_data}")
# enc_data :
# [[0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
print(f"type_data : {type(_data)}")     # type_data : <class 'numpy.ndarray'>  
print(f"_data.dtype : {_data.dtype}")   # _data.dtype : float64

#1-3. 데이터 슬라이싱
x_data = _data[:6, ]    # hihell
y_data = _data[1:, ]    # ihello

print(f"x_data: \n{x_data}")
# x_data:
# [[0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]]
print(f"y_data: \n{y_data}")
# y_data:
# [[0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
# print(f"x_data.shape: {x_data.shape}")

#1-4. y데이터 shape 복귀
# y데이터는 6개를 출력하여 값을 비교해야함
# 케라스에서는 (6,)로 진행했지만, 텐서플로우 1버전에서는 (1,6)로 진행해야함
# 원핫인코딩을 하기 전으로 y데이터를 되돌림
y_data = np.argmax(y_data, axis=1)
print(f"y_data : \n {y_data}")
# y_data :
#  [2 1 0 3 3 4]
print(f"y_data.dtype : {y_data.dtype}")     # y_data.dtype : int64  
print(f"y_data.shape : \n {y_data.shape}")  # y_data.shape : (6,)

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)
print(f"3D x_data : \n {x_data}")
# x_data : # (1,6,5)
#  [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 1. 0. 0. 0.]
#   [1. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 1. 0.]]]
print(f"2D y_data : \n {y_data}")
# y_data :  # (1,6)
#  [[2 1 0 3 3 4]]
print(f"y_data.dtype : {y_data.dtype}")     # y_data.dtype : int64  

#1-5. shape 부분을 변수명으로 저장
seq_length = 6
input_dim = 5
output = 5
batch_size = 1  # 전체 행

#1-6. feed_dict에 feed될 placeholer 생성
# x = tf.placeholder(tf.float32, (None,sequence_length,input_dim))
# y = tf.placeholder(tf.float32, (None,sequence_length))
x = tf.compat.v1.placeholder(tf.float32, (None,seq_length,input_dim))
y = tf.compat.v1.placeholder(tf.int32, (None,seq_length))

#2. 모델 구성
# model.add(LSTM(output, input_shape=(6,5)))
# 중간 계산 과정을 생각한 cell 구성
cell = tf.nn.rnn_cell.BasicLSTMCell(output)
h, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print(f"h: {h}")    # (?, 6, 100)

#3. 컴파일 : 손실 함수 정의 / 실제값-예측값의 차이를 최소화
w = tf.ones([batch_size, seq_length])    
# 디폴트로 선형을 그리면서 모델 훈련을 하겠다
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=h, targets=y, weights=w)
# logits=각 열의 클래스와 일치하는 pred(예측값) 여기서는 h를 지정
# targets=실제 y값 지정
cost = tf.reduce_mean(seq_loss)     
# seq_loss의 전체 평균

#3-1. 옵티마이저 정의
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

#4. 예측
# (1,6,100) 100 부분을 pred로 지정해야하므로 axis=2
pred = tf.argmax(h, axis=2)
print(pred)     # (?, 6)

#5. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        res = sess.run(pred, feed_dict={x:x_data})
        print(i, f"loss: {loss}, pred: {res}, true_y : {y_data}")
        
        # print(np.squeeze(res))  #[2 1 0 3 3 4]

        res_str = [idx2char[c] for c in np.squeeze(res)]
        print(f"pred str\n: {res_str}")
