'''
--url--
https://www.acmicpc.net/problem/5086

--title--
5086번: 배수와 약수

--problem_description--
4 × 3 = 12이다.

이 식을 통해 다음과 같은 사실을 알 수 있다.

3은 12의 약수이고, 12는 3의 배수이다.

4도 12의 약수이고, 12는 4의 배수이다.

두 수가 주어졌을 때, 다음 3가지 중 어떤 관계인지 구하는 프로그램을 작성하시오.

--problem_input--
입력은 여러 테스트 케이스로 이루어져 있다. 각 테스트 케이스는 10,000이 넘지않는 두 자연수로 이루어져 있다. 
마지막 줄에는 0이 2개 주어진다. 두 수가 같은 경우는 없다.

--problem_output--
각 테스트 케이스마다 첫 번째 숫자가 두 번째 숫자의 약수라면 factor를, 배수라면 multiple을, 둘 다 아니라면 neither를 출력한다.

'''

#1. while문으로 마지막 줄에 0 0이 출력되기 전까지 테스트 케이스 입력받기
#2. 입력받은 a,b,c가 1)if b%a==0: factor출력 2)elif a%b==0: multiple출력 3)else: neither 출력

import sys

a,b = map(int,sys.stdin.readline().split())

while (a!=0) & (b!=0):

    if b%a==0:
        print("factor")
    elif a%b==0:
        print("multiple")
    else:
        print("neither")
    a,b = map(int,sys.stdin.readline().split())
