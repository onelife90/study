# 리스트 변수를 다른 변수에 대입한 뒤 변수에서 값을 바꾸면 원래 변수의 값도 변함

X = ["a","b","c"]
X_copy = X
X[0] = "A"
print(X)        # ['A', 'b', 'c']

# 이를 방지하기 위해서 x_copy = x[:] 또는 x_copy = list(x)라고 수정

c = ["red","blue","yellow"]

# 변수 c의 값이 변하지 않도록 수정
c_copy = c[:]
c_copy[1] = "green"
print(c)        # ['red', 'blue', 'yellow']
