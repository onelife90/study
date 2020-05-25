# 확률(probability) 어떠한 사건의 공간에서 특정 사건이 선택될 때 발생하는 불확실성을 수치적으로 나타내는 것
# 사건 E. 확률 P(E) 표현

# 사건 E의 발생 여부가 사건 F의 발생 여부에 대한 정보를 제공한다면 두 사건 E,F는 종속사건 <-> 독립사건
# 사건 E와 F가 동시에 발생할 확률이 각각 사건이 발생할 확률의 곱과 같다면 두 사건은 독립 사건

# 조건부확률. 사건 F가 일어났을 때 사건 E가 발생할 확률
# P(E,F) = P(E)*P(F)
# P(E|F) = P(E,F)/P(F)

# ex1) 공이 들어있는 주머니. 
# 주머니 A = 빨간공이 7개, 파란공이 3개 / 주머니 B = 빨간 공이 5개, 파란 공이 5
# 이 때 어느 주머니인지는 모르나 빨간공이 나왔음. 주머니 A에서 빨간공이 나올 확률을 구할 때 조건부 확률 사용
# P(A | 빨간공) = P(A) P (A & 빨간공) / P(빨간공)

# ex2) 철수와 영희. 철수=전화, 영희=문자.
# 스마트폰 문자 도착. 그 문자가 철수에게서 온 것인지 영희에게서 온 것인지 조건부 확률로 구할 수 있음 

# ex3) 한 가족 안의 두 아이의 성별 맞추기
# 조건1_각 아이가 딸이거나 아들일 확률은 동일
# 조건2_둘째의 성별은 첫째의 성별과 독립
# 첫째가 딸인 경우(사건 G), 두 아이가 모두 딸일 (사건 B) 확률?
# P(B|G) = P(B,G)/P(G) = P(B)/P(G)= 1/4 // 1/2 == 1/2

# ex4) 딸이 최소 한 명인 경우(사건L), 두 아이가 모두 딸일(사건B) 확률 계산
# P(B|L) = P(B,L)/P(L)=P(B)/P(L)= 1/4 // 3/4 == 1/3
# 두 아이가 모두 딸이 아닌 경우=1/4의 반대 경우이므로 P(L)=3/4

class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger=random_kid()
    older=random_kid()
    if older==Kid.GIRL:
        older_girl +=1
    if older==Kid.GIRL and younger==Kid.GIRL:
        both_girls +=1
    if older == Kid.GIRL or younger==Kid.GIRL:
        either_girl +=1

print("P(both | older):", both_girls/older_girl)    # 0.514 ~ 1/2
print("P(both | either):", both_girls/either_girl)  # 0.342 ~ 1/3
