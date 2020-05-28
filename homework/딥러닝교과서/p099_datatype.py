h = 1.7
w = 60
# 변수 h, w 자료형 출력
print(type(h))
print(type(w))
# 변수 bmi에 비만도 지수 계산 결과 대입
bmi = w/h**2
# 변수 bmi 출력
print(bmi)
# 변수 bmi의 자료형 출력
print(type(bmi))

h = 1.7
w = 60
bmi = w/h**2
# '당신의 bmi는 OO입니다.'라고 출력
print('당신의 bmi는' + str(bmi) + '입니다.')        # 형변환 필수

n = "10"
print(n*3)      # 101010, str형
