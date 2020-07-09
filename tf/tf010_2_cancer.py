# 이진분류

from sklearn.datasets import load_breast_cancer
import tensorflow as tf

#1. 데이터
cancer = load_breast_cancer()
x_data = cancer.data
y_data = cancer.target
# print(x_data.shape) # (569, 30)
# print(y_data.shape) # (569,)

y_data = y_data.reshape(-1,1)
# print(y_data.shape) # (569,1)

#1-1. feed_dict에 feed 될 텐서를 위한 placeholder 설정
x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

#2. 모델 구성
w = tf.Variable(tf.random_normal([30,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))
h = tf.nn.sigmoid(tf.matmul(x,w) + b)

#2-1. cost 손실함수(binary_crossentropy) 정의
cost = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))

#2-2. loss를 최소화하는 옵티마이저 정의
opt = tf.train.GradientDescentOptimizer(learning_rate=5e-3)
train = opt.minimize(cost)

#3. 평가, 예측 : h>0.5이면 예측을 하겠다
# 천천히보자
#1) tf.equal(pred,y) : pred와 y값을 비교하여 float32의 형태로 캐스팅
#2) tf.reduce_mean(캐스팅한 값)을 전체 평균 내어 모든 차원이 제거되고 단 하나의 스칼라 값이 출력 
pred = tf.cast(h>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

#3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 5000번의 epoch로 훈련되는 과정
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
    
    # 케라스로 따지면 최적의w, y_pred, acc 출력
    # 훈련 과정에 포함 되지 않아도 되므로 if문 구역 밖& for문 안에 위치
    h,c,a = sess.run([h, pred, acc], feed_dict={x:x_data, y:y_data})
    print("H: \n", h, "\n correct(y): \n", c, "\n acc: ", a)
    # acc:  0.37258348
