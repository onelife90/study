# 딕셔너리형 루프에서는 키와 값을 모두 변수로 하여 루프가능
# for key_변수명, value의 변수명 in 변수(딕셔너리형).items()

town = {"경기도": "분당", "서울":"중랑구", "제주도":"제주시"}
# for문을 사용하여 출력
for a, b in town.items():
    print(a,b)
# 경기도 분당
# 서울 중랑구
# 제주도 제주시
