# --title--
# 5543번: 상근날드

# --problem_description--
# 상근날드에서 가장 잘 팔리는 메뉴는 세트 메뉴이다. 주문할 때, 자신이 원하는 햄버거와 음료를 하나씩 골라, 세트로 구매하면, 가격의 합계에서 50원을 뺀 가격이 세트 메뉴의 가격이 된다.
# 햄버거와 음료의 가격이 주어졌을 때, 가장 싼 세트 메뉴의 가격을 출력하는 프로그램을 작성하시오.

# --problem_input--
# 입력은 총 다섯 줄이다. 첫째 줄에는 상덕버거, 둘째 줄에는 중덕버거, 셋째 줄에는 하덕버거의 가격이 주어진다. 넷째 줄에는 콜라의 가격, 다섯째
#  줄에는 사이다의 가격이 주어진다. 모든 가격은 100원 이상, 2000원 이하이다.

# --problem_output--
# 첫째 줄에 가장 싼 세트 메뉴의 가격을 출력한다.

#1. 버거, 음료 가격 입력받기
s = int(input())
j = int(input())
h = int(input())
coke = int(input())
cider = int(input())

'''
burger = [s, j, h]
# 버거 세 개중 하나 + 콜라 or 사이다
for i in range(len(burger)):
    if burger[i] + coke < burger[i] + cider:
        set = 
'''
#2. 다 때려박은 리스트 에서 인덱스 슬라이싱으로 최소값 뽑기
set = [s, j, h, coke, cider]
print(min(set[0:3])+min(set[3:])-50)
