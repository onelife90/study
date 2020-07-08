# x,y를 정제된 데이터 말고 placeholder를 쓸 수 있지 않을까?

import tensorflow as tf
# 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

# random_normal 정규분포로부터의 난수값 반환 ([1])=shape
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

print(w)

w = tf.Variable([0.3], tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(w)

print("w : ", aaa)
# w :  [0.3]
sess.close()

# 변수에 바로 InteractiveSession을 쓰면 .eval을 해주면됨
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = w.eval()
print("InteractiveSession을 한 w: ",bbb)
# InteractiveSession을 한 w:  [0.3]
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session=sess)
print("eval에 session 명시한 w: ", ccc)
# eval에 session 명시한 w:  [0.3]
sess.close()
