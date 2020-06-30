# 10039 평균 점수 문제
# 문제
# 다섯 사람 수강생 시험 채점. 40점 미만인 학생들은 항상 40점 부여. 학생 5명의 점수가 주어졌을 때 평균 점수를 구하는 프로그램 작성

#- 입력 
# 총 5줄로 이루어져 있고, 원섭이의 점수, 세희의 점수, 상근이의 점수, 숭이의 점수, 강수의 점수가 순서대로 주어진다.
# 점수는 모두 0점 이상, 100점 이하인 5의 배수이다. 따라서, 평균 점수는 항상 정수이다. 
#- 출력
# 첫째 줄에 학생 5명의 평균 점수를 출력한다.

#1. 입력 총 5줄이기에 input 5줄
s1 = int(input())
s2 = int(input())
s3 = int(input())
s4 = int(input())
s5 = int(input())

#1-1. sys.stdin.readline().split() 사용해보기 : 런타임에러
# import sys
# s1,s2,s3,s4,s5 = map(int, sys.stdin.readline().split())

#2. 5개의 점수를 리스트 형태로 묶기
score=[s1,s2,s3,s4,s5]

#3. for문 안에 if문을 사용하여 40점 미만이면 40점으로 리스트 안 수정하고, 평균 출력 : 런타임에러
#3-1. 구글링 결과 파이썬 리스트 함수 중에 average가 없음 : 합계/5
#3-2. print(sum(score)/5)으로 하니 실수로 출력이 되어서 int 형변환
for i in range(5):
    if score[i]<40:
        score[i]=40
    # print(score)
print(int(sum(score)/5))


'''
a = [10,20,30,40,50]
# print(sum(a)) # 150

for i in range(5):
    if a[i]<40:
        a[i]=40
        print(a)
        print(sum(a)/5)

# 리스트와 정수 비교 (결과:에러)
# TypeError: argument of type 'int' is not iterable
# TypeError: '<' not supported between instances of 'list' and 'int'
'''
