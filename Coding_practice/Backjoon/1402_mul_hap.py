'''
--url--
https://www.acmicpc.net/problem/1402

--title--
1402번: 아무래도이문제는A번난이도인것같다

--problem_description--
6 = 2*3, 2+3=5이므로 성립한다.

이때 A와 B가 주어지면 A는 B로 변할 수 있는지 판별하시오.

--problem_input--
1
6 5

--problem_output--
각각의 테스트 케이스마다 한 줄에 변할 수 있으면 yes, 아니면 no를 출력한다.

'''

#1. test_case 입력받기
#2. for 문 내에 split해서 a,b 받기
#3. a%2=0, b=2+(a//2) / (a%2)!=0, b=1+a

import sys

test = int(sys.stdin.readline())

for t in range(test):
    a,b = map(int, sys.stdin.readline().split())
    if a%2==0:
        a1 = 2
        a2 = a//2
        b = 2+(a//2)
        print("yes")
    elif a%2!=0:
        a1 = 1
        a2 = 1*a
        b = 1+a
        print("yes")
    else:
        print("no")
