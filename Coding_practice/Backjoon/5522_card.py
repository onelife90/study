# 5522 문제
# JOI군은 카드 게임을 하고 있다. 이 카드 게임은 5회의 게임으로 진행되며, 그 총점으로 승부를 하는 게임이다.
# JOI군의 각 게임의 득점을 나타내는 정수가 주어졌을 때, JOI군의 총점을 구하는 프로그램을 작성하라.

# 입력
# 표준 입력에서 다음과 같은 데이터를 읽어온다.
# i 번째 줄(1 ≤ i ≤ 5)에는 정수 Ai가 적혀있다. 이것은 i번째 게임에서의 JOI군의 점수를 나타낸다.
# 모든 입력 데이터는 다음 조건을 만족한다.
# 0 ≤ Ai ≤ 100．

# 출력 : 표준 출력에 JOI군의 총점을 한 줄로 출력하라.

# 생각 프로세스
#1. map input()메서드로 그냥 프린트 : 런타임 에러
# a1,a2,a3,a4,a5 = map(int, input().split())
# sum = a1+a2+a3+a4+a5
# print(sum)

#2. ai == a1,a2,a3,a4,a5가 되어야 하는데 type(a)==str, type(i)==int
#3. a를 int로 형변환하면 되지 않을까? 응 안돼

# sum=0
# for i in range(1,6):
#     # print(type(i))
#     ai = map(int, input().split())
#     sum += ai
# print(sum)

#4. game이라는 리스트에 a1,a2,a3,a4,a5를 넣어두고 수를 입력받아 총점을 구하자 : int와 map의 타입에러
# game = ["a1","a2","a3","a4","a5"]
# hap = 0
# for i in range(5):
#     game[i] = map(int, input().split())
#     hap += int(game[i])
# print(hap)

#5. 다시 문제를 보자 i번째 줄(1<=i<=5)이므로 5개의 input이 필요 / 해결!
a1 = int(input())
a2 = int(input())
a3 = int(input())
a4 = int(input())
a5 = int(input())

sum = a1+a2+a3+a4+a5
print(sum)
