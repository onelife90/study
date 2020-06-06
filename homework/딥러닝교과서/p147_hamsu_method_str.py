# 메서드 : 어떠한 값에 대해 처리를 하는 것
# 값.메서드형()
# 함수는 ()안에 기입, 메서드는 값 뒤에 .을 연결해 기술
# .append : 리스트형에 쓰는 메서드
# sorted(변수명) / 변수명.sort()
# sorted 함수는 변수의 내용을 변경 X
# .sort() 메서드는 변수의 내용까지 변경 파괴적 메서드

# 문자열형 메서드
# .upper() : 모든 문자열을 대문자로 반환
# .count() : ()안에 들어있는 문자열에 요소가 몇개인지 반환
animal = "elephant"
# 변수 animal에 저장되어 있는 문자열을 대문자로 변환해서 변수 anmail_big에 저장
animal_big = animal.upper()
print(animal)       # elephant
print(animal_big)   # ELEPHANT
# 변수 animal에 'e'가 몇 개 포함되어 있는지 출력
print(animal.count('e'))        # 2

# fomat() : 임의의 값을 대입한 문자열 생성. 문자열에 변수를 삽입할 때 자주 사용
# 문자열 내에 {}를 포함하는 것이 특징
fruit = "바나나"
color = "노란색"
# 바나나는 노란색입니다 출력
print("{}는 {}입니다.".format(fruit, color))        # 바나나는 노란색입니다.
