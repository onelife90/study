# 가장 중요한 리스트 슬라이싱
# 리스트에서 새로운 리스트 추출

chaos =["cat", "apple", 2, "orange", 4, "grape", 3, "banana", 1, "elephant", "dog"]

# chaos 리스트에서 ["apple",2,"orange",4,"grape",3,"banana",1] 리스트를 꺼내 변수 fruits에 저장
fruits = chaos[1:-2]
print(fruits)       # ['apple', 2, 'orange', 4, 'grape', 3, 'banana', 1]
