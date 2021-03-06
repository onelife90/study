'''
--url--
https://www.acmicpc.net/problem/10872

--title--
10872번: 팩토리얼

--problem_description--
0보다 크거나 같은 정수 N이 주어진다
이때, N!을 출력하는 프로그램을 작성하시오.

--problem_input--
첫째 줄에 정수 N(0 ≤ N ≤ 12)가 주어진다.

--problem_output--
첫째 줄에 N!을 출력한다.

'''
# 첫번째 방법
#1. 정수 입력받기
#2. 파이썬에서 제공하는 팩토리얼 함수 사용
#3. math import 후 factorial() 메서드 사용

import sys
import math

n = int(sys.stdin.readline())

print(math.factorial(n))

# 두번째 방법
#1. 정수 입력받기
#2. for문 사용하여 1부터 n까지 곱하기

fac = 1
for i in range(1,n+1):
    fac *=  i
print(fac)
