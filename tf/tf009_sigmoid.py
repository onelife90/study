# h에 활성화 함수 사용해보자

import tensorflow as tf

#1-1. 그래프 수준의 난수 시드 설정
tf.set_random_seed(777)

#1-2. 데이터 / y데이터를 보면 다중분류
x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

#2. 모델 구성
# random_normal 정규분포로부터의 난수값 반환
# w, b는 인풋과 아웃풋 값을 기반으로 변경이 되며 인공지능은 바로 w를 예측하는 것이 핵심이기에 변수로 선언하여 구현
# 행렬 곱 연산 : (6,2)*(6,1)==(2,1)
w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 행렬 곱 연산해주는 matmul을 랩핑해서 sigmoid 활성화 함수 사용
# sigmoid는 엄밀히 말하면 linear regressor(다중분류!)
h = tf.sigmoid(tf.matmul(x, w) + b)


#2-1. cost 손실함수 binary_crossentropy 정의
cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

#2-2. cost를 최소화하는 옵티마이저 정의. minimize(cost)=cost가 가장 적을 때 구하라
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)

#3. 평가, 예측 : h>0.5이면 예측을 하겠다
# tf.cast는 텐서를 새로운 형태로 캐스팅하는데 사용
pred = tf.cast(h>0.5, dtype=tf.float32)

# 천천히보자
#1) tf.equal(pred,y) : pred와 y값을 비교하여 float32의 형태로 캐스팅
#2) tf.reduce_mean(캐스팅한 값)을 전체 평균 내어 모든 차원이 제거되고 단 하나의 스칼라 값이 출력 
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

#4. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 5000번의 epoch로 훈련되는 과정
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step%10==0:
            print(step, "cost: ", cost_val)

    # 케라스로 따지면 최적의w, y_pred, acc 출력
    # 훈련 과정에 포함 되지 않아도 되므로 if문 구역 밖& for문 안에 위치
    h, c, a = sess.run([h, pred, acc], feed_dict={x:x_data, y:y_data})
    print("H : \n", h, "\n correct(y) : \n", c ,"\n ACC : ", a)
