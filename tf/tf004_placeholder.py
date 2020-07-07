'''
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# session을 통과하지 않았으므로 자료형 출력
# print("node3 : ", node3)
# node3 :  Tensor("Add:0", shape=(), dtype=float32)
# Add = node1 + node2
'''
# placeholder : 변수와 비슷한 개념이긴 하지만, 더 정확히는 인풋과 비슷한 개념. sess.run을 할 때 넣어줌
# sess.run(feed_dict= ) 꼭 나와줘야 인풋의 개념으로 적용

import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# adder_node이라는 그래프
adder_node = a+b

print("a+b : ", sess.run(adder_node, feed_dict={a:3, b:4.5}))
# a+b :  7.5

print("a+b : ", sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))
# a+b :  [3. 7.]

# add_and_triple이라는 다른 그래프
add_and_triple = adder_node*3
print("a*3: ", sess.run(add_and_triple, feed_dict={a:3, b:4.5}))
# a*3:  22.5

# 그림으로 표현하자면, 딥러닝의 가중치 연산과 비슷. 즉, 텐서플로우 session는 그래프를 그리는 연산
