#2. 튜플
# 리스트와 거의 같으나 삭제, 수정이 안된다 ==> 고정값에 쓸 수 있음
# ex) 게임 전사캐릭터
a = (1,2,3)
b = 1,2,3
print(type(a))
print(type(b))

# a.remove(2)     # 에러
# print(a)

print(a+b)      #(1, 2, 3, 1, 2, 3)
print(a*3)      #(1, 2, 3, 1, 2, 3, 1, 2, 3)   # a*3==a+a+a

# print(a-3)    # 에러
