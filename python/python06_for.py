# 컴퓨터는 단순 연산함 ex)3*5=3+3+3+3+3 다만 그 속도가 빠를 뿐
# 조건문과 반복문 - 이것만 잘해도 코딩 잘한다는 소리 들음

a = {'name':'yun', 'phone':'010', 'birth':'0511'}       #키 값에는 정수형, 문자형 다 사용가능

for i in a.keys():          #파이썬에서 : enter를 치면 자동으로 탭이 됨==그 구역에 포함
    print(i)                #name #phone #birth     
    # for문은 반복이기 때문에 결과값이 하나씩 출력
    # i 말고 다른 변수 사용 가능 --> for문을 쓰면서 바로 정의가 되기 때문에

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:                 # a라는 리스트 인자 수만큼 돌려라 즉, 10번 돌려짐
    i = i*i
    print(i)
    # print('melong')         #melong이 10번 출력. for문 구역에 포함됐기 때문에
# print('melong')               #melong이 1번 출력. for문 구역 밖

for i in a:
    print(i)

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
                    # 이대로 돌리면 출력이 됨    # 이 조건을 판단하는 것이기 때문에

if 3:
    print('True')
else:
    print('False')  

if 0:               # 0==거짓, 1==참
    print('True')
else:
    print('False') 

if -1:
    print('True')
else:
    print('False')   

'''
비교연산자

<, >, ==, !=, >=, <=

'''

a = 1                   #a라는 변수를 선언해야 출력됨
if a == 1:              #a=1이라고 쓰면 syntax 문법 에러
    print('출력잘돼')   

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')

# 딥러닝에서 if문 나오면 1)조건파악 2)if문 구역이 어디까지 포함인지 체크

### 조건연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card ==1:
    print('한우먹자')
else:
    print('라면먹자')

jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:                     #for문은 다섯번 돌아감_jumsu 리스트 인자가 5개라서
    if i >= 60 :
        print("경] 합격 [축")
        number = number + 1

print("합격인원 : ", number, "명")

##########################################################
# break, continue
print("==================break==================")
jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:                   
    if i < 30:
        break                       # break에서 걸리면 break와 가장 가까운 for문을 날려버림
    if i >= 60 :
        print("경] 합격 [축")
        number = number + 1

print("합격인원 : ", number, "명")

print("==================continue==================")
jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:                   
    if i < 60:
        continue            #continue의 조건이 맞으면 하단 부분 for문을 실행하지 않고 처음으로 돌아감
    if i >= 60 :
        print("경] 합격 [축")
        number = number + 1

print("합격인원 : ", number, "명")
