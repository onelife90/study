# 람다는 def로 정의한 함수와 달리 반환값 부분에 식만 넣을 수 있음
# lambda 인수 : 반환값
# if 조건문에서는 삼항 연산자를 이용하여 람다를 작성 가능
# 조건을 먼저 생각하고 다음 효과를 생각하는 통상적인 사고 흐름에 반하는 것
# '조건을 만족할 경우의 처리 if 조건 else 조건을 만족하지 않을 경우의 처리'

a1 = 13
a2 = 32

func5 = lambda x : x**2-40*x+350 if 10<=x<30 else 50
print(func5(a1)) # -1
print(func5(a2)) # 50
