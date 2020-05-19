# 정수형
a = 1
b = 2
c = a+b
print(c)
d=a*b
print(d)
e = a / b
# 파이썬은 정수/정수=실수 _ C언어나 자바는 정수/정수=정수
print(e) #0.5

#실수형
a=1.1
b=2.2
c=a+b
print(c) #3.3000000000000003
# 파이썬은 실수를 부동소수점 방식으로 표현. 하지만 정확히 표현할 수 없다
# 컴퓨터에서는 숫자를 비트로 표현
# ex) 5.0e-4 = 0.0005으로 끝에 자리를 잘라버림
# 64비트끼리 더하면 끝에 떨거지들이 생길 수 밖에 없음

d=a*b
print(d) #2.4200000000000004

e=a/b
print(e) #0.5

# 문자형
a = 'hel'
b = 'lo'
c = a+b
print(c)

# 문자 + 숫자
a = 123
b = '45'
# c = a+b # 타입 에러
# print(c)

#### 형변환 ####
# 숫자를 문자 변환 + 문자
a = 123
a = str(a)
print(a)
b = '45'
c = a+b
print(c)

# 문자를 숫자로 변환
a = 123
b = '45'
b = int(b)
c = a+b
print(c)

# 문자열 연산하기
a = 'abcdefgh'
print(a[0])
print(a[3])
print(a[5])
print(a[-1]) # 마이너스면 거꾸로
print(a[-2])
print(type(a)) # <class 'str'> 
# .shape와 엄청 많이 사용

b = 'xyz'
print(a+b)

# 문자열 인덱싱
a = 'Hello, Deep learning'
print(a[7]) #D  # 띄어쓰기도 문자
print(a[-1]) #g
print(a[-2]) #n
print(a[3:9]) #lo, De

# :-x == 뒤에서부터 x번째 '전'까지
print(a[3:-5]) #lo, Deep lea
print(a[:-1]) #Hello, Deep learnin
print(a[1:]) #ello, Deep learning
print(a[5:-4]) #, Deep lear
