# 2742번: 기찍 N

# --problem_description--
# 자연수 N이 주어졌을 때, N부터 1까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.

# --problem_output--
# 첫째 줄부터 N번째 줄 까지 차례대로 출력한다.

#1. 자연수 n 입력받기
n = int(input())
# print()
# a = range(100)
# print(len(a))   # 100

#2. n부터 1까지 한 줄씩 출력되어야 하므로 for문 사용

''' 실패작 컬렉션

# 분석해보겠다
# 첫번째 쓰렉
for i in range(1,n+1):
    if i==1:
        print(n)
    elif i==n:
        print(1)
    n -= 1
    print(n)

# n=5라 가정
# i=1, n=5, n=4 (for문 안에 아직 print(n)이 존재하기 때문)
# i=2, n=3(n-=1의 실행)
# i=3, n=1(elif 조건문 만족), n=2(n-=1의 실행)
# i=4, n=1(n-=1의 실행)
# i=5, n=0(n-=1의 실행)

# 두번째 쓰렉
for i in reversed(range(n+1)):
    i==n
    print(n)
    if i==0:
        break
# n=5라 가정
# 당근 reversed했으니 i==n에서 5가 6번 출력

### 두번째 쓰렉에서 정답을 발견
# reversed 하고 i만 출력하면 됨
for i in reversed(range(1,n+1)):
    print(i)
'''

new = n+1
for i in range(1,n+1):
    new -= 1
    print(new)
