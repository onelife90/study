for x in range(10):
    if x == 3:
        continue    # x==3이면 다 날려버림
    if x == 5:
        break       # x==5이면 다 멈춰라
    print(x)
