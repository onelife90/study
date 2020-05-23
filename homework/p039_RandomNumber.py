import random

random.seed(10)         # seed를 10으로 설정
print(random.random())  # 0.5714025946899135
random.seed(10)         # seed를 다시 10으로 설정해도
print(random.random())  # 0.5714025946899135가 출력

up_to_ten = [1,2,3,4,5,6,7,8,9,10]
random.shuffle(up_to_ten)
print(up_to_ten)
