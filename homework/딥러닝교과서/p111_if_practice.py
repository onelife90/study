# 윤년 여부를 검사하는 프로그램 작성
#1) 연도가 400으로 나누어 떨어지지 않고, 100으로 나누어 떨어지면 윤년입니다(평년)
#2) 위와 같이 않을 때 연도가 4로 나누어떨어지면 윤년입니다
#3) 윤년이 아닌 경우 평년
#4) 윤년의 경우 '**년은 윤년입니다'라고 출력
#5) 평년의 경우 '**년은 평년입니다'라고 출력

# 변수 year에 연도를 입력하세요
year = 2020

# if 문으로 조건 분기를 하고 윤년과 평년을 판별하세요
if year%400 !=0 and year%100==0:
    print(year, "년은 평년입니다.")
elif year%4==0:
    print(year, "년은 윤년입니다.")
else:
    print(year, "년은 평년입니다.")

# 같지않다 !=
# 같다 == 반드시 2개 쓸 것!
