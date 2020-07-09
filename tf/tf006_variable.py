import tensorflow as tf

#1-1. 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

#1-2. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

#2. 모델 구성
# random_normal 정규분포로부터의 난수값 반환. ([1])=shape
# w, b는 인풋과 아웃풋 값을 기반으로 변경이 되며 인공지능은 바로 w를 예측하는 것이 핵심이기에 변수로 선언하여 구현
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# print(w)    <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

#2-1. session 통과 방법(1) : session -> 변수 초기화 -> run 
w = tf.Variable([0.3], tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

aaa = sess.run(w)
print("w : ", aaa)
# w :  [0.3]
sess.close()

#2-2. session 통과 방법(2) : InteractiveSession -> 변수 초기화 -> eval(run 없이 통과)
# InteractiveSession을 쓰면 자기 자신을 기본 세션으로 설치, .eval()과 .run 메서드는 연산을 실행하기 위해 자기 자신의 세션 사용
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# bbb라는 변수명은 sess.run 없이 통과 가능
bbb = w.eval()
print("InteractiveSession을 한 w: ",bbb)
# InteractiveSession을 한 w:  [0.3]
sess.close()

#2-3. session 통과 방법(3) : session -> 변수 초기화 -> eval(session=sess) 명시
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session=sess)
print("eval에 session 명시한 w: ", ccc)
# eval에 session 명시한 w:  [0.3]
sess.close()
