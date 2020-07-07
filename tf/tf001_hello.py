import tensorflow as tf
print("tf.__version__", tf.__version__) # tf.__version__ 1.14.0

# constant는 텐서플로우에서 operation이라 부르고 일반적으로 함수라고 생각
# operation들은 텐서 객체를 출력으로 제공해주는 주체
# operation들은 session 안에서만 실행이 되며 session 밖에서는 정의가 되면서 그래프를 생성할 뿐
hello = tf.constant("Hello AI")

print("hello 텐서의 자료형: ", hello)  # Tensor("Const:0", shape=(), dtype=string)

# Session : 우리가 결과물을 눈에 보고 싶으면 session을 통과시켜야 함
# 머신이 장치 안에서 연산을 하면 더 수월하게 잘 작동하기 때문에
# input할 때 규격에 맞게 넣어줘야 함
# 저 머신을 꼭 통과해야함? 초심자들에게는 불필요. 그래서 나온 것이 케라스. 즉, 머신이 backend가 됨
sess = tf.Session()
print("hello의 session 통과 후 출력: ", sess.run(hello))
# hello의 session 통과 후 출력:  b'Hello AI'
# 'Hello AI' 앞에 있는 b는 바이트 문자열로 이루어진 텐서를 의미
