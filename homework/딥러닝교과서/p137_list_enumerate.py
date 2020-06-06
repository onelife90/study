# for문에서 index 표시
# for문을 사용한 루프에서 리스트의 인덱스 확인이 필요할 때가 있다
# enumerate() 함수를 사용하여 인덱스가 포함된 요소를 얻을 수 있음
# for x,y in enumerate("리스트형"):
#   for 안에서는 x,y 를 사용하여 작성
#   x=정수형 인덱스, y=리스트에 포함된 요소
# x,y는 인덱스와 요소를 얻기 위한 변수. 자유롭게 변수명 수정 가능

animals = ["tiger", "dog", "elephant"]
# enumerate() 함수를 사용하여 출력
for index, values in enumerate(animals):
    print(index, values)
# 0 tiger
# 1 dog
# 2 elephant
