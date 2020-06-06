# 리스트의 요소가 리스트형일 경우 그 내용을 for문으로 꺼낼 수 있음
# for a,b,c,.. in 변수(리스트형)
# 이 때 a,b,c..의 개수는 리스트의 요소수와 같아야함!

fruits = [["strawberry", "red"],
          ["peach", "pink"],
          ["banana", "yellow"]]
# for문을 사용하여 출력
for index, values in fruits:
    print(index + " is "+ values)
# strawberry is red
# peach is pink
# banana is yellow

# 리스트 안 , 콤마 필수! 콤마 없으니 tuple로 인식
