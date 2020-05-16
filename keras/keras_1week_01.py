#1. 딥러닝 시작
# h(x) = wx+b
# 딥러닝 : 우리가 빅데이터로 준비한 x(input)과 y(pouput)값을 가지고 컴퓨터에 훈련 시켜서 w와 bias를 구하는 행위의 반복
# 머신의 cost가 낮을수록 좋다. loss 그래프 연상

# 딥러닝에서 가장 쉬운 케라스 사용
# from keras.(models, layers 등) import -- / 이렇게 간단히 API호출
# 신셩망 - 동그란 부분 node, 각 층을 layer
# 일반적으로 노드가 많고 레이어가 깊을수록 더 훈련을 잘하나, 과적합의 문제
# 결국 모델을 생성시, 얼마나 많은 레이어와 노드를 준비할 것인지에 대해 설계하는 게 우리의 일

# compile = 머신이 어떤 방식으로 모델을 돌릴 것인지에 대해 지정해주는 일
# loss(손실 함수), optimizer(최적화 함수), metrics(판정 방식)의 옵션으로 지정

# model.fit=x와 y를 fitness 센터에 보내서 훈련시키겠다
# epoch=몇 번 훈련을 시킬지, batch_size=몇 개씩 끊어서 작업
# model.evaluate=최종 결과에 대한 평가

#2. 모델 구성
# train(훈련용), test(평가용) data 정의
# model = Sequential() 순차적 모델 구성
# model.add(Dense(5, input_dim=1)) dimension 차원 / 1개의 입력을 받아 5개의 노드로 출력
# 수정가능한 부분? 노드, 레이어의 깊이
# model.summary() 모델 구성 확인 함수

# parameter=(input노드+bias노드(1))*output노드
# 머신은 딥러닝 할 때 바이어스도 1개의 노드로 계산
# model.fit에 validation data가 추가 / 머신에게 훈련데이터와 평가데이터를 나눠서 학습과 평가하기 위해서
# 왜 나누는가? 수능시험 답만 외운 애들은 실제 수능에서 망함. 즉, 평가데이터는 모델에 반영 X

# 딥러닝 케라스의 기본 구조
# 1)데이터준비 2)모델 구성 3)컴파일,훈련 4)평가,예측

# 네이밍룰 : 강제적이진 않으나 암묵적인 룰
# Java 카멜케이스 ex)whiteSnowPrincess / 변수(소문자)+뜻이 변하는 부분에서 대문자
# Keras 언더바
# 나중에 클래스 개념-대문자 / WhiteSnowPricess(Java), White_snow_princess(Keras)
