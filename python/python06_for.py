# 컴퓨터는 단순 연산함 ex)3*5=3+3+3+3+3 다만 그 속도가 빠를 뿐
# 조건문과 반복문 - 이것만 잘해도 코딩 잘한다는 소리 들음

a = {'name':'yun', 'phone':'010', 'birth':'0511'}       
# dict형     # {key:value}   # 키 값에는 정수형, 문자형 다 사용가능

for i in a.keys():          # 파이썬에서 ':enter'를 치면 자동으로 탭이 됨==그 구역에 포함
    print(i)                # name  # phone  # birth     
    # for문은 반복이기 때문에 keys가 하나씩 출력
    # i 말고 다른 변수 사용 가능 --> 그럼 왜 i를? for문을 쓰면서 바로 정의가 되기 때문에

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:                 # a라는 리스트 인자 수(10)만큼 for문이 반복
    i = i*i
    print(i)                # 1\n 4\n 9\n 16\n 25\n 36\n 49\n 64\n 81\n 100   
                            # 실제 출력은 개행을 넣은 것처럼 하나씩 출력. 왜? for문이 10번 반복되어 결과값 출력
    # print('melong')       # melong이 10번 출력. for문 구역 포함
# print('melong')           # melong이 1번 출력. for문 구역 밖

for i in a:
    print(i)                # 1\n 2\n 3\n 4\n 5\n 6\n 7\n 8\n 9\n 10
                            # 이 또한 for문이 10번 반복되어 결과값 10개 출력

## while문
'''
while 조건문 :      # 참일 동안 계속 돈다
    수행할 문장
'''

### if문

if 1:
    print('True')
else:
    print('False')
                    # True   # 이대로 돌리면 출력이 됨. 왜? 입력값이 없는데?   # 이 조건 자체를 판단하는 것이기 때문에 1이면 True

if 3:
    print('True')
else:
    print('False')  
                    # True
        
if 0:               # 0==거짓, 1==참
    print('True')
else:
    print('False')  # False

if -1:      
    print('True')
else:
    print('False')   
                    # True
        
'''
비교연산자

<, >, ==, !=, >=, <=

'''

a = 1                   # a라는 변수를 선언해야 출력됨
if a == 1:              # a=1이라고 쓰면 syntax 문법 에러
    print('출력잘돼')   

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')   # 라면먹자

# 딥러닝에서 if문 나오면 1)조건파악 2)if문 구역이 어디까지 포함인지 체크

### 조건연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card ==1:
    print('한우먹자')
else:
    print('라면먹자')   # 한우먹자

jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:                     #for문은 다섯번 돌아감_jumsu 리스트 인자가 5개라서
    if i >= 60 :
        print("경] 합격 [축")       # "경] 합격 [축" X3==3번출력
        number = number + 1

print("합격인원 : ", number, "명")   # 합격인원 :3명

##########################################################
# break, continue
print("==================break==================")
jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:                   
    if i < 30:
        break                       # break에서 걸리면 break와 가장 가까운 for문을 날려버림
    if i >= 60 :
        print("경] 합격 [축")        # "경] 합격 [축" ==1번출력 / break에 걸린 for문이 날아가버림
        number = number + 1

print("합격인원 : ", number, "명")   # 합격인원:1명

print("==================continue==================")
jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:                   
    if i < 60:
        continue                   # continue의 조건이 맞으면 하단 부분 for문을 실행하지 않고 처음으로 돌아감
    if i >= 60 :
        print("경] 합격 [축")       # "경] 합격 [축" X3
        number = number + 1

print("합격인원 : ", number, "명")   # 합격인원:3명
