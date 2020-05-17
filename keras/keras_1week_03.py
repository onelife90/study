#3. 회귀모델
# 딥러닝은 크게 2개의 모델로 나눈다. 1)회귀 모델 2)분류모델
# 1)회귀모델 : 수치를 넣었을 때 수치로 답한다
# 2)분류모델 : 강아지(0), 고양이(1) 값이 정해져있다 ex)동전을 던지고 난 결과
# 회귀 모델 y=wx+b가 기본. 평가 지표가 acc라면, 엉뚱한 값에도 결과가 부여 ex) 강아지(0), 고양이(1), 공룡(0,xxx)
# 회귀 분석은 선형이기 때문에 딱 맞아떨어지는 값이 아님. 그래서 결과값(model.evaulate)과 예측 분석(model.predict)을 하는 함수를 다른것으로 사용

#4. 회귀모델의 판별식
# RMSE(평균 제곱근오차) : 각종 데이터대회, 케글에서 많이 모델을 평가하는 지표
# 사이킷런에는 아직 RMSE가 없어서, MSE를 함수로 호출하여 루트를 씌우자
# (실제값-예측값)^2 다 더해서/n에 루트를 씌운 것 = RMSE / 루트를 씌웠기 때문에 낮을수록 정밀도가 높다

# RMSE와 회귀 분석에서 가장 많이 쓰이는 R2 지표
# R2, R2 score, R제곱, 설명력, 결정계수 등 여러가지 이름. RMSE와 반대로 높을수록 좋은 지표. max값=1
# 만약 R2 값이 음수가 나왔다면, 학습 시 머신에 뭔가 잘못된 부분이 있을 수 있다는 의미

# validation 추가
# model.fit에서 검증값을 test로 하면 훈련셋이 검증값이 들어가고 그 검증값으로 다시 테스트를 한다 --> 평가에 검증이 반영이 되는 문제
# train(교과서), validation(모의고사)로 나누어서 model.fit에 훈련하면서 검증을 하면 됨 / test(model.evaluate)
# 즉, train 데이터의 일부를 잘라서 validation 데이터로 사용 권장

# def RMSE(y_test, y_predict): 에서 예측값과 테스트값의 오차를 비교하기 위해 입력값에 y_test, y_predict를 넣어줌
# R2도 마찬가지로 입력값에 y_test, y_predict를 넣어서 비교

# 데이터 분리
# [:60] = 0번째 인덱스 ~ 60번째 인덱스 전까지
# [60:80] = 60번째 인덱스 ~ 80번째 인덱스 전까지
# [80:] = 80번째 인덱스 ~ 끝까지

#train_test_split