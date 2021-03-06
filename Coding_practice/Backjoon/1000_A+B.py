# 문제 : 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

## 풀이
# 생각 프로세스
#1. 두 정수 A,B를 입력받기 : 여기서 입력을 '받는'에 의미가 있다고 생각해서 "정수 입력 받기 파이썬" 구글링
#2. A, B = int(input().split()) 처음 시도는 틀렸다고 나오기에
#3. 정수 입력 받는 다른 코드 구글링
#4. A, B = map(int, input().split()) 맞음

# A, B = map(int, input().split())
# print(A+B)


# 밑에 주석은 (0 < A, B < 10)의 조건에 맞게 하기 위해 짜봤던 거지만 너무 많은 오류에 시간을 허비할 수 없기에 일단 패스
# try-except 사용
# if문 사용
'''
try:
    A, B = map(int, input().split())
if A|B < 0:
    except IndexError:
    print("0보다 작은 수 입니다. 다시 입력해주세요 : ")
except IndexError:
    print("10보다 큰 수입니다. 다시 입력해주세요 : ")

print(A+B)

while A <= 0:
    print(input("A는 0보다 큰 정수입니다. 다시 입력해주세요 : "))
    if A > 0:
        break
#         B = int(input("10보다 작은 정수를 입력해주세요 : "))
#         print(B)
# # else:
#     B = int(input("10보다 작은 정수를 입력해주세요: "))

if B > 10:
    print(input("B는 10보다 작은 정수입니다. 다시 입력해주세요 : "))

print(A+B)

# A = int(input("0보다 큰 정수를 입력해주세요 : "))
# B = int(input("10보다 작은 정수를 입력해주세요: "))

# print(A+B)
'''
