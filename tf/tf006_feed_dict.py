# 텐서플로우는 그래프를 미리 만들어 놓고, 필요한 시점에 해당 그래프를 실행하는 지연실행 방식을 사용
# 이에 걸 맞게 placeholder는 변수의 타입을 미리 설정해놓고 필요한 변수를 나중에 받아서 실행

import tensorflow as tf

#1-1. 그래프 수준의 난수 시드 설정
# random_seed는 (graph, operation)의 두개 시드를 도출함. 여기서는 그래프 시드 도출. seed는 파이썬 정수
tf.set_random_seed(777)

#1-2. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
# placeholder로 y_train에는 실수형에 텐서의 구조가 어떤지 모르는 값을 세팅
# 어떤 인풋 값에 따라서 아웃풋 내용이 달라지기 때문에 값을 받아들이는 통로라고 생각하자

#2. 모델 구성
# random_normal 정규분포로부터의 난수값 반환. ([1])=shape
# w, b는 인풋과 아웃풋 값을 기반으로 변경이 되며 인공지능은 바로 w를 예측하는 것이 핵심이기에 변수로 선언하여 구현
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print("w의 가중치: ", sess.run(w))
# w의 가중치:  [2.2086694]

h = x_train * w + b

#2-1. cost 손실 함수 정의
# 케라스 컴파일 시, loss=(cost)를 줄여나가는 과정
cost = tf.reduce_mean(tf.square(h - y_train))
# (h-y)**2의 합 / 개수 == mse

#2-2. cost를 최소화하는 옵티마이저 정의. minimize(cost)=cost가 가장 적을 때 구하라
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#3. 훈련
# with 문 안에 sess가 포함
# with tf.Session() as sess:
with tf.compat.v1.Session() as sess:
    # 전체 변수들이 싹 초기화
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        # _:공백(실행은 되지만 결과값 출력 안하겠다)
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step%20==0:
            print(step, cost_val, w_val, b_val)

    # 예측해보자
    print("4의 예측값: ", sess.run(h, feed_dict={x_train:[4]}))
    # 4의 예측값:  [9.000078]
    print("[5,6]의 예측값: ", sess.run(h, feed_dict={x_train:[5,6]}))
    # [5,6]의 예측값:  [11.000123 13.000169]
    print("[6,7,8]의 예측값: ", sess.run(h, feed_dict={x_train:[6,7,8]}))    
    # [6,7,8]의 예측값:  [13.000169 15.000214 17.000257]
