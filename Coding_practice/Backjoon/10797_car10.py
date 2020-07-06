'''
--url--
https://www.acmicpc.net/problem/10797

--title--
10797번: 10부제

--problem_description--
서울시는 6월 1일부터 교통 혼잡을 막기 위해서 자동차 10부제를 시행한다. 
자동차 10부제는 자동차 번호의 일의 자리 숫자와 날짜의 일의 자리 숫자가 일치하면 해당 자동차의 운행을 금지하는 것이다. 
예를 들어, 자동차 번호의 일의 자리 숫자가 7이면 7일, 17일, 27일에 운행하지 못한다. 
또한, 자동차 번호의 일의 자리 숫자가 0이면 10일, 20일, 30일에 운행하지 못한다.
여러분들은 일일 경찰관이 되어 10부제를 위반하는 자동차의 대수를 세는 봉사활동을 하려고 한다.
날짜의 일의 자리 숫자가 주어지고 5대의 자동차 번호의 일의 자리 숫자가 주어졌을 때 위반하는 자동차의 대수를 출력하면 된다. 

--problem_input--
첫 줄에는 날짜의 일의 자리 숫자가 주어지고 두 번째 줄에는 5대의 자동차 번호의 일의 자리 숫자가 주어진다. 날짜와 자동차의 일의 자리 숫자는 모두 0에서 9까지의 정수 중 하나이다. 

--problem_output--
주어진 날짜와 자동차의 일의 자리 숫자를 보고 10부제를 위반하는 차량의 대수를 출력한다.

'''

#1. 날짜 끝번호와 자동차 끝번호가 일치하는 조건
#2. 날짜와 자동차 5개를 비교하여 일치하는 것의 개수 출력
#3. 자동차 하나하나씩 요소를 꺼내와서 비교해야 하므로 리스트에 담자
#4. 첫번째 방법 : 함수 설정 / 메모리(29380KB), 시간(60ms) 소요 큼
#5. 두번째 방법 : 리스트 컴프리헨션 / 메모리(29380KB), 시간(60ms) 소요 큼
#6. 더 좋은 방법 있을 거 같은뎅

import sys
date = int(sys.stdin.readline())
c1,c2,c3,c4,c5 = map(int, sys.stdin.readline().split())

cars = [c1,c2,c3,c4,c5]

# 첫번째 제출 : 함수
def vio(x):
    return x==date

vio_cars = list(filter(vio,cars))
print(len(vio_cars))

# 두번째 제출 : 리스트 컴프리헨션
# Nested List Comprehension은 조건을 추가하여 iterable에서 사용할 item(원소)를 선별
vio_cars = [car for car in cars if car==date]
print(len(vio_cars))