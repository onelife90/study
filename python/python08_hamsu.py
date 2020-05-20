def sum1(a, b):     # def=함수 정의, sum1=함수명, (a,b)=입력값, 매개변수명(a,b) 변경가능
    return a+b      

a = 1
b = 2
c = sum1(a,b)       

print(c)            # 3

### 곱셈, 나눗셈, 뺄셈 함수를 만드시오.
# mul1, div1, sub1
def mul1(a,b):
    return a*b
print(a*b)          # 2

def div1(a,b):
    return a/b
print(a/b)          # 0.5

def sub1(a,b):
    return a-b
print(a-b)          # -1

def sayYeh():       # 매개변수(파라미터)가 없는 함수
    return 'Hi' 

aaa = sayYeh()
print(aaa)          # sayYeh

def sum1(a, b, c):      #def=함수 정의, sum1=함수명, (a,b)=입력값, 매개변수명 변경가능
    return a+b+c

a = 1
b = 2
c = 34
d = sum1(a,b,c)

print(d)            # 37
