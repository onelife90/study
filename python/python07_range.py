# range함수(클래스)
a = range(10)
print(a)            # range(0,10)   # [0,1,2,3,4,5,6,7,8,9]     # 0번째 인덱스 ~ 10번째 인덱스 '전'까지  
b = range(1, 11)    
print(b)            # range(1,10)   # [1,2,3,4,5,6,7,8,9,10]    # 1번째 인덱스 ~ 11번째 인덱스 '전'까지

for i in a: 
    print(i)        # 0\n 1\n 2\n 3\n 4\n 5\n 6\n 7\n 8\n 9     # 개행을 넣은 것처럼 10개의 결과값 도출
for i in b:
    print(i)        # 1\n 2\n 3\n 4\n 5\n 6\n 7\n 8\n 9\n 10    # 10개의 결과값 도출

print(type(a))      # <class 'range'>

sum = 0
for i in range(1,11):
    sum = sum + i   # 1부터 10까지의 총합
print(sum)          # 55
