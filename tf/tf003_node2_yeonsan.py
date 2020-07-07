# 텐서플로우 1대 버전 사칙연산을 해보자
# 3+4+5
# 4-3
# 3*4
# 4/2

import tensorflow as tf

node0 = tf.constant(2.0)
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)

def s_run(input):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = input.eval()
        return(res)

# tf.add_n은 많은 양의 텐서를 한번에 처리. []로 묶어줘야 처리 가능, tf.add = (x,y) 2가지만 처리
print("3+4+5 : ", s_run(tf.add_n([node1, node2, node3])))
# 3+4+5 :  12.0
print("4-3 : ", s_run(tf.subtract(node2, node1)))
# 4-3 :  1.0
print("3*4 : ", s_run(tf.multiply(node1, node2)))
# 3*4 :  12.0
print("4/2 : ", s_run(tf.divide(node2, node0)))
# 4/2 :  2.0
