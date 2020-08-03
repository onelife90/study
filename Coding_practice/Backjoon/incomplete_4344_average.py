'''
--url--
https://www.acmicpc.net/problem/4344

--title--
4344번: 평균은 넘겠지

--problem_description--
대학생 새내기들의 90%는 자신이 반에서 평균은 넘는다고 생각한다
당신은 그들에게 슬픈 진실을 알려줘야 한다.

--problem_input--
첫째 줄에는 테스트 케이스의 개수 C가 주어진다.
둘째 줄부터 각 테스트 케이스마다 학생의 수 N(1 ≤ N ≤ 1000, N은 정수)이 첫 수로 주어지고, 이어서 N명의 점수가 주어진다
점수는 0보다 크거나 같고, 100보다 작거나 같은 정수이다.

--problem_output--
각 케이스마다 한 줄씩 평균을 넘는 학생들의 비율을 반올림하여 소수점 셋째 자리까지 출력한다.

'''

# 5
# 5 50 50 70 80 100
# 7 100 95 90 80 70 60 50
# 3 70 90 80
# 3 70 90 81
# 9 100 99 98 97 96 95 94 93 91

#1. c를 입력받고, for문으로 case 세트 반복
#2. 필요한 것 : 점수, 평균, 평균 이상인 점수의 개수 비율

import sys

# score = map(int, sys.stdin.readline().split())
# ave = sum(score)/len(score)
# print(score)    # map 자료형 출력

c = int(sys.stdin.readline())

for case in range(c):
    n = int(sys.stdin.readline())
    for scores in range(n):
        score = map(int, sys.stdin.readline().split())
        ave = sum(score)/n
print()
