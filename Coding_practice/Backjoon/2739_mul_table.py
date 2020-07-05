'''
--url--
https://www.acmicpc.net/problem/2739

--title--
2739번: 구구단

--problem_description--
N을 입력받은 뒤, 구구단 N단을 출력하는 프로그램을 작성하시오. 출력 형식에 맞춰서 출력하면 된다.

--problem_input--
첫째 줄에 N이 주어진다. N은 1보다 크거나 같고, 9보다 작거나 같다.

--problem_output--
출력형식과 같게 N*1부터 N*9까지 출력한다.

'''

#1. 정수 n 입력받기
#2. n*1~9 한줄에 하나씩 구구단 연산 출력해야하므로 for문 작성
#3. 구구단의 곱은 str으로 출력. f-string 사용
#4. f-string 쓰니까 틀렸다고 해서 프린트문에 str 형식으로 바로 넣음

import sys
n = int(sys.stdin.readline())

for i in range(1,10):
    print(n,"*",i,"=", n*i)
    #print(f'{n}*{i} =',n*i)

'''리스트 컴프리헨션 (리스트 형태로 출력)
import sys
import numpy as np

n = int(sys.stdin.readline())
mul = [n*m for m in range(1,10)]
print(mul)
'''
