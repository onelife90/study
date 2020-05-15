# 데이터 두개 이상을 사용해보자
# ex) data = 삼성, 하이닉스 주가 output = 다우지수, xx지수 (output은 2개 이상 나올 수 있음)


#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) # 100개짜리 3덩어리
y = np.array([range(101,201), range(711,811), range(100)])
# 파이썬에는 list가 있는데 덩어리를 []로 묶어줘야함
# 1st 덩어리 : w=1, b=100
# 2nd 덩어리 : w=1, b=400
# 3rd 덩어리 : w=1

x = np.transpose(x)
y = np.transpose(y)
# 처음엔 np.transpose(x)만 코딩하고 왜 안되지 함
# 안되는 이유? transpose한 값을 x라는 공간에 저장을 하지 않았기 때문! 즉, 변수의 초기화 단계는 필수
print(x.shape)
print(y.shape)

# (3,100) 3행 100열 / 뭔가 이상? 3행이면 가로 3줄
# 통상적인 엑셀 기입방법을 떠올려보자 : 데이터를 추가할 때(↓)는 행에 추가가 됨 
# x(컬럼) = 날씨, 용돈, 삼성 / y = sk
# 외워라~ 열우선, 행무시
# input_dim=3, 3개 컬럼을 사용하겠다 / 다른 x 데이터 종류 +1 추가하려면 에러가 뜸. 왜? input_dim=3이기 때문에
# (3,100) 3행 100열 이기 때문에 우리가 통상적으로 생각하는 엑셀로 바꿔줘야함
