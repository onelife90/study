'''
--problem_description--
정수 n개가 주어졌을 때, n개의 합을 구하는 함수를 작성하시오.

def solve(a: list) -> int
a: 합을 구해야 하는 정수 n개가 저장되어 있는 리스트 (0 ≤ a[i] ≤ 1,000,000, 1 ≤ n ≤ 3,000,000)
리턴값: a에 포함되어 있는 정수 n개의 합 (정수)

'''

#1. solve라는 함수 정의
#2. a=[] n=정수의 개수
#3. 리스트의 합
#4. return solve
#5. 입력 받는 것이 아님? for문으로 연속 합 구하기

# import sys

# n = list(map(int, sys.stdin.readline().split()))
# n = map(int, sys.stdin.readline().split())
# a = []

def solve(a):
    # a.append(n)
    res = 0
    for i in a:
        res += i
    return res
