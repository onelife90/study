# lr가 작으면 그라디언트 소실
# lr이 높으면 그라디언트 배니싱
# 방지하는 방법을 알아보자

w = 0.5
input = 0.5
goal_pred = 0.8     # 얘를 찾아가는 과정

lr = 0.001

for i in range(1101):
    pred = input*w
    error = (pred-goal_pred)**2

    print("error: " + str(error) + "\tPred : " + str(pred))

    # goal_pred를 찾아가다가 너무 높게 잡아서 지나치는 기준
    up_pred = input*(w+lr)
    up_error = (goal_pred - up_pred)**2

    # goal_pred를 찾아가다가 너무 낮게 잡아서 지나치는 기준
    down_pred = input*(w-lr)
    down_error = (goal_pred-down_pred)**2

    # 그 기준이 너무 크다면 w -= lr 해서 다시 goal_pred를 찾아가라
    if(down_error < up_error):
        w -= lr

    # 그 기준이 너무 낮다면 w += lr 해서 다시 goal_pred를 찾아가라
    if(down_error > up_error):
        w += lr
