# h = wx+b
# aaa, bbb, ccc 자리에 각 h를 구하시오

import tensorflow as tf
# 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

# random_normal 정규분포로부터의 난수값 반환 ([1])=shape
x = [1.,2.,3.]
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([1], tf.float32)
h = w*x + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(h)
print("h : ", aaa)
# w :  [0.3]
sess.close()

# 변수에 바로 InteractiveSession을 쓰면 .eval을 해주면됨
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = w.eval()
print("InteractiveSession을 한 h: ",bbb)
# InteractiveSession을 한 w:  [0.3]
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session=sess)
print("eval에 session 명시한 h: ", ccc)
# eval에 session 명시한 w:  [0.3]
sess.close()
