'''
--url--
https://www.acmicpc.net/problem/2523

--title--
2523번: 별 찍기 - 13

--problem_description--
예제를 보고 규칙을 유추한 뒤에 별을 찍어 보세요.

--problem_input--
3
--problem_output--
*
**
***
**
*

'''
#1. 첫번째 줄 1개 -> n번째 줄까지 1개씩 늘어남
#2. n번째+1 줄 -> 끝번째 줄까지 1개씩 줄어듬

# i = 5
# print("*"*i)    # *****

import sys
n = int(sys.stdin.readline())

for i in range(n):
    print("*"*i)

for j in range(n-1):
    print("*"*(n-j-1))
