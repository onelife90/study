# loss 함수 그래프 식으로 나타내면, y=ax**2+b+c
# y=wx+b를 구하는 과정 : 최적의 가중치, 최소의 loss(cost)
# 미분하여 최적의 가중치 구하기
# lr이 너무 작으면 그라디언트 손실이 발생 / lr이 너무 크면 그라디언트 vanish 폭주가 발생

# lambda : 간략한 함수
# return이 없이 바로 들어감
gradient = lambda x: 2*x - 4
# 적분 x**2-4x+b

def gradient2(x):
    temp = 2*x - 4
    return temp

x = 3

print(gradient(x))      # 2
print(gradient2(x))     # 2
