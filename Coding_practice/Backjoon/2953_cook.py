'''
--url--
https://www.acmicpc.net/problem/2953

--title--
2953번: 나는 요리사다

--problem_description--
	"나는 요리사다"는 다섯 참가자들이 서로의 요리 실력을 뽐내는 티비 프로이다. 
	각 참가자는 자신있는 음식을 하나씩 만들어오고, 서로 다른 사람의 음식을 점수로 평가해준다. 점수는 1점부터 5점까지 있다.
	각 참가자가 얻은 점수는 다른 사람이 평가해 준 점수의 합이다. 이 쇼의 우승자는 가장 많은 점수를 얻은 사람이 된다.
	각 참가자가 얻은 평가 점수가 주어졌을 때, 우승자와 그의 점수를 구하는 프로그램을 작성하시오.

--problem_input--
	총 다섯 개 줄에 각 참가자가 얻은 네 개의 평가 점수가 공백으로 구분되어 주어진다. 
	첫 번째 참가자부터 다섯 번째 참가자까지 순서대로 주어진다. 항상 우승자가 유일한 경우만 입력으로 주어진다.

--problem_output--
	첫째 줄에 우승자의 번호와 그가 얻은 점수를 출력한다.

'''

#1. for문을 이용하여 입력 총 다섯줄 받기
#2. 네가지 변수 split로 입력받기
#3. 각 줄의 점수를 리스트로 만들기
#4. 점수 리스트 중에서 가장 큰 index와 총합 출력

import sys
import numpy as np

score = []

for i in range(5):
	a,b,c,d = list(map(int,sys.stdin.readline().split()))
	# print(a,b,c,d)
	# print(type(a))
	sum = a+b+c+d
	score.append(sum)
print(score.index(max(score))+1,max(score))
