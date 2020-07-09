# h = wx+b
# aaa, bbb, ccc 자리에 각 h를 구하시오

import tensorflow as tf

#1-1. 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)
x = [1,2,3]

#2. 모델 구성
# random_normal 정규분포로부터의 난수값 반환.
# w, b는 인풋과 아웃풋 값을 기반으로 변경이 되며 인공지능은 바로 w를 예측하는 것이 핵심이기에 변수로 선언하여 구현
# random_normal을 빼줘야 실행되는 이유는? w와 b의 값을 우리가 지정을 해줬기 때문에 필요없다
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)
h = w*x + b

#2-1. session 통과 방법(1) : session -> 변수 초기화 -> run 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(h)
print("h : ", aaa)
# w :  [0.3]
sess.close()

#2-2. session 통과 방법(2) : InteractiveSession -> 변수 초기화 -> eval(run 없이 통과)
# InteractiveSession을 쓰면 자기 자신을 기본 세션으로 설치, .eval()과 .run 메서드는 연산을 실행하기 위해 자기 자신의 세션 사용
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = w.eval()
print("InteractiveSession을 한 h: ",bbb)
# InteractiveSession을 한 w:  [0.3]
sess.close()

#2-3. session 통과 방법(3) : session -> 변수 초기화 -> eval(session=sess) 명시
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session=sess)
print("eval에 session 명시한 h: ", ccc)
# eval에 session 명시한 w:  [0.3]
sess.close()
