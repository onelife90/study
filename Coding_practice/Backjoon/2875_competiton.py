'''
--url--
https://www.acmicpc.net/problem/2875

--title--
2875번: 대회 or 인턴

--problem_description--
백준대학교에서는 대회에 나갈 때 2명의 여학생과 1명의 남학생이 팀을 결성해서 나가는 것이 원칙이다.
 (왜인지는 총장님께 여쭈어보는 것이 좋겠다.)

백준대학교는 뛰어난 인재들이 많아 올해에도 N명의 여학생과 M명의 남학생이 팀원을 찾고 있다. 
대회에 참여하려는 학생들 중 K명은 반드시 인턴쉽 프로그램에 참여해야 한다. 
인턴쉽에 참여하는 학생은 대회에 참여하지 못한다.

백준대학교에서는 뛰어난 인재들이 많기 때문에, 많은 팀을 만드는 것이 최선이다.

여러분은 여학생의 수 N, 남학생의 수 M, 인턴쉽에 참여해야하는 인원 K가 주어질 때 만들 수 있는 최대의 팀 수를 구하면 된다.

--problem_input--
첫째 줄에 N, M, K가 순서대로 주어진다. (0 ≤ M ≤ 100, 0 ≤ N ≤ 100, 0 ≤ K ≤ M+N),

--problem_output--
만들 수 있는 팀의 최대 개수을 출력하면 된다.

'''

#1. 사람을 기준으로 생각하니 너무 많은 조건이 생성
#2. 컨닝
#3. team을 초기화
#4. 팀 조건 추가 : n은 2명 이상, m은 1명 이상, n+m>= k+3(최소 1팀 결성 인원+인턴)

import sys

n,m,k = map(int, sys.stdin.readline().split())

# team을 초기화
team = 0
while (n>=2) & (m>=1) & (n+m>=k+3):
    n -= 2
    m -= 1
    team += 1
print(team)


'''
if n>m:
    team = (n-k)//2
elif n<m:
    n = n-(n%2) 
    team = n//2
elif n==m!=k:
    team = n//2
else:
    team=0

# team = (n+m-k)//3
print(team)
'''
