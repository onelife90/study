A = {"지우개":[100,2], "펜":[200,3], "노트":[400,5]}
total_price = 0

# 변수 items를 for문으로 루프
for i in A:
    # '**은 한 개에 **원이며, **개 구입합니다.'라고 출력
    print(i + "은(는) 한 개에"+ str(A[i][0]) + "원이며" + str(A[i][1]) + "개 구입합니다")
    # 변수 total_price에 가격X수량을 더해서 저장
    total_price += A[i][0] * A[i][1]
# 지불해야 할 금액은 '**원 입니다.'라고 출력
print("지불해야 할 금액은" + str(total_price) + "입니다.")

# 변수 money에 임의의 값을 대입하시오
money = 2000
# money>total_price일 때는 '거스름돈은 **원입니다'라고 출력
if money>total_price:
    print("거스름돈은" + total_price - money + "원입니다")
# money == total_price일 때는 '거스름돈은 없습니다' 출력
elif money == total_price:
    print("거스름돈은 없습니다")
# money<total_price일 때는 '돈이 부족합니다'라고 출력
else:
    print("돈이 부족합니다")

# 지우개은(는) 한 개에100원이며2개 구입합니다
# 펜은(는) 한 개에200원이며3개 구입합니다
# 노트은(는) 한 개에400원이며5개 구입합니다
# 지불해야 할 금액은2800입니다.
# 돈이 부족합니다
