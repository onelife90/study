'''
--url--
https://www.acmicpc.net/problem/10950

--title--
10950번: A+B - 3

--problem_description--
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

--problem_input--
첫째 줄에 테스트 케이스의 개수 T가 주어진다.

각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

--problem_output--
각 테스트 케이스마다 A+B를 출력한다.

'''

#1. 5번 반복. input 받아서 range로 iterable하게 만들어야함
#2. for 문 안에 input(spilt())
#3. print(a+b)

import sys

tests = range(int(sys.stdin.readline()))

for test in tests:
    a,b = map(int, sys.stdin.readline().split())
    print(a+b)
