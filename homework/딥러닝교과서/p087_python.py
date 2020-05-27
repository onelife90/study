print("Hello AI")

# 5+2의 결과를 출력
print(5+2)
# 3+8의 결과 출력
print(3+8)

# 숫자 18 출력
print(18)
# 2+6 계산 후 출력
print(2+6)
# '2+6'이라는 문자열 출력
print('2+6')

# 3+5
print(3+5)
# 3-5
print(3-5)
# 3x5
print(3*5)
# 3/5
print(3/5)
# 3%5   3을 5로 나눈 나머지
print(3%5)
# 3의 5제곱
print(3**5)

# 변수n에 '고양이' 대입
n = "고양이"
# 변수n 출력
print(n)
# 'n'이라는 문자열 출력
print('n')
# 변수 n에 3+7이라는 수식 대입
n = 3+7
# 변수 n을 출력
print(n)

m = "고양이"
print(m)
# 변수 m의 값을 '강아지'로 덮어쓰고 출력
m = '강아지'
print(m)
n = 14
print(n)
# 변수 n에 5를 곱하여 덮어쓰세요
n *= 5      # n = n*5
print(n)

# 변수 p에 '서울' 대입
p = '서울'
# 변수 p를 사용하여 '나는 서울 출신입니다.' 출력
print('나는'+ p + '출신입니다.')

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
print('당신의 bmi는' + bmi + '입니다.')

n = "10"
print(n*3)      # 101010, str형

# !을 이용해서 4+6과 -10의 관계식을 만들어 True를 출력
print(4+6 != -10)

n = 16
# if문을 사용하여 n이 15보다 크면 '큰 숫자'를 출력
if n>15:
    print("큰 숫자")

# else를 사용하여 n이 15보다 작으면 "작은 숫자"를 출력
n = 14
if n>15:
    print("큰 숫자")
else:                   # 조건을 입력할 필요 X. if의 반대 경우이기 때문에
    print("작은 숫자")

# elif를 사용하여 n이 11이상 15 이하일 때 '중간 숫자'를 출력
n = 14
if n>15:
    print("큰 숫자")
elif 11<=n<=15:
    print("중간 숫자")
else:
    print("작은 숫자")

n_1 = 14
n_2 = 28
# n_1이 8보다 크고 14보다 작다는 조건식을 만들고 결과 출력
print(8<n_1<14)

# n_1의 제곱이 n_2의 다섯 배보다 작다는 조건식을 만들고 결과 반전
