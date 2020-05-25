# 리스트나 제너레이터에서 항목을 하나씩 확인해 볼 경우, 항목의 순서를 반환하고 싶을 때
# enumerate 함수를 사용하면 (순서, 항목) 형태로 값을 반환

names = ["Alice", "Bod", "charlie", "Debbie"]

for i, name in enumerate(names):
    print(f"name {i} is {name}")
