# while문은 조건식이 False가 될 때까지 처리하는 반복
# while조건식 : 

x = 5
while x>0:
    print("Hanbit")
    x -= 2

# Hanbit을 몇번이나 출력할까?
# x=5, print1
# x=3, print2
# x=1, print3

x = 5
# while문을 사용하여 변수 x가 0이 아닌 동안 반복
while x!=0:
    x -= 1
    print(x)
# 4
# 3
# 2
# 1
# 0
# 내가 실수한점. print문이 while문 구역 밖에 있어서 0만 출력됨. 조심하소서

x = 5
# while문을 사용하여 변수 x가 0이 아닌 동안 반복
while x!=0:
    x -= 1
    if x != 0:
        print(x)
    else:
        print("Bang")
# 4
# 3
# 2
# 1
# Bang
