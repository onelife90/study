# 리스트[인덱스 번호]=값을 사용하면 지정한 인덱스 번호의 요소를 수정
# 리스트에 요소를 추가하고 싶은 경우 리스트와 리스트를 '+'를 사용하여 연결
# 리스트명.append(추가할 요소)

c = ["dog", "blue", "yellow"]

# 변수 c의 첫 번째 요소룰 'red'로 수정
c[1] = "red"
print(c)         # ['dog', 'red', 'yellow']

# 리스트 끝에 문자열 'green'을 추가
c.append("green")
print(c)        # ['dog', 'red', 'yellow', 'green']
