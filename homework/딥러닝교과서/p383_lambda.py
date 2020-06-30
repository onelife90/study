# 파이썬에서 함수를 작성할 때 def를 사용하지만 익명 함수(lambda식)를 사용하여 코드를 좀 더 간단히 만듬
# 람다식의 구조 lambda 인수: 반환값

a=4
def func1 (x):
    return 2*x**2-3*x+1

func2 = lambda x: 2*x**2-3*x+1

print(func1(a))     # 21
print(func2(a))     # 21

# 람다에 의한 계산
# 다변수 함수(둘 이상의 독립변수를 갖는 함수)를 작성
# 변수에 저장하지 않고도 사용 가능
#ex) print((lambda x,y: x+y)(3,5))
# 하지만 이 방법은 람다식을 사용한 의미X

x=5
y=6
z=2

def func3 (x,y,z):
    return x*y+z

func4 = lambda x,y,z: x*y+z

print(func3(x,y,z))     # 32
print(func4(x,y,z))     # 32
