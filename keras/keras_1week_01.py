#1. 딪러닝 시작
# h(x) = wx+b
# 딥러닝 : 우리가 빅데이터로 준비한 x(input)과 y(pouput)값을 가지고 컴퓨터에 훈련 시켜서 w와 bias를 구하는 행위의 반복
# 머신의 cost가 낮을수록 좋다. loss 그래프 연상

# 딥러닝에서 가장 쉬운 케라스 사용
# from keras.(models, layers 등) import -- / 이렇게 간단히 호출
# 신셩망 - 동그란 부분 node, 각 층을 layer
# 일반적으로 노드가 많고 레이어가 깊을수록 더 훈련을 잘하나, 과적합의 문제
# 결국 모델을 생성시, 얼마나 많은 레이어와 노드를 준비할 것인지에 대해 설계하는 게 우리의 일

# compile = 머신이 어떤 방식으로 모델을 돌릴 것인지에 대해 지정해주는 일
# loss(손실 함수), optimizer(최적화 함수), metrics(판정 방식)의 옵션으로 지정

# model.fit=x와 y를 fitness 센터에 보내서 훈련시키겠다
# epoch=몇 번 훈련을 시킬지, batch_size=몇 개씩 끊어서 작업
# model.evaluate=최종 결과에 대한 평가
