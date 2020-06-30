#1. 리스트 분할(하나의 기호로 분할)
# 문자열을 공백이나 슬래시 등으로 분할할 때는 split()함수 사용
# 나눌 문자열.split("구분기호", 분할 횟수)

self_data = "My name is Jieun"
split = self_data.split(" ")
print(split[3]) # Jieun

#2. 리스트 분할(여러 기호로 분할)
# 표준 split()함수는 한 번에 여러 기호로 분할 불가능
# re모듈에 포함된 re.split()함수 사용
# re.split("[구분기호]", 분할할 문자열)

import re

time_data = "2020/6/30_15:12"
time = re.split("[/_:]", time_data)
print(time[1])  # 6
print(time[3])  # 15
