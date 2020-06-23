# 문제
# 전화번호 중, 한 번호가 다른 번호의 접두어인 경우가 있는지 확인
# phone_book = ["119", "97674223", "1195524421"] False
# phone_book = ["123", "456", "789"] True
# phone_book = ["12", "123", "1235", "567", "88"] False

'''
a = ["119","97674223","1195524421"]
# print(a[1]-a[0])      # string 에러
print(a.remove("119"))  # None
# print(a[1]-"119")     # string 에러
print(a.remove(a[0]))   # None
'''
b = [1,2,3, 'a', 'b']
print(type(b[3]))   # <class 'str'>


## 내 풀이 ##
phone_book = ["12", "123", "1235", "567", "88"]
phone_book.sort()   # 오름차순
print(phone_book)
# print(type(phone_book[0]))  # <class 'str'>
# print(len(phone_book))      # 3

for i in range(5):
    for j in range(5):
        print(i,j)
# 두번째 for문부터 돌고 첫번째 for문으로 감

# 여기서.. phone_book은 세가지 리스트 이상이어야 함
def solution(phone_book):
    phone_book.sort()
    if (phone_book[0] in s for s in phone_book):
        return False
    return True
# ㄴㄴ 이것은 쓰레기..
# 당연히 0번째 있는 것을 있냐고 물어보는 for문

### 해민스님 풀이###

def sol(values):#index 0~3
    pre = values[0]#119
    for i in values[1:]:#1191, 12321, 132424
        if pre == i[:len(pre)] and len(pre)<=len(i):# (119 == i[0:3] <-"119") and  (3<=i의 길이)
            return False
    return True

# for i in range() ==> i는 숫자가 오고
# for i in list or str ==> i는 리스트 요소나 str이 온다

print(sol(["119","1191"]))
