import tensorflow as tf
# 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [3,5,7]

# random_normal 정규분포로부터의 난수값 반환 ([1])=shape
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print("w의 가중치: ", sess.run(w))
# w의 가중치:  [2.2086694]

h = x_train * w + b

# 케라스 컴파일 시 가장 중요한 것 loss=(cost)
cost = tf.reduce_mean(tf.square(h - y_train))
# (h-y)**2의 합 / 개수 == mse

# 케라스 컴파일 시 옵티마이저(경사하강법). minimize(cost)=cost가 가장 적을 때 구하라
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# with 문 안에 sess가 포함
with tf.Session() as sess:
    # 전체 변수들이 싹 초기화
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        # _:공백(실행은 되지만 결과값 출력 안하겠다)
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b])
        if step%20==0:
            print(step, cost_val, w_val, b_val)
