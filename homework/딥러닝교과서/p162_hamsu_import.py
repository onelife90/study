# 파이썬에서는 일반에 공개된 함수를 import하여 사용
# 유사한 용도끼리 셋으로 공개. 이 셋을 패키지. 패키지 안에 있는 하나하나의 함수를 모듈
# 패키지를 import하면 '패키지명.모듈명'으로 함수 사용
# 'from 패키지명 import 모듈명'으로 모듈을 import하면 패키지명 생략하고 사용 가능
# ex) from sklearn.model_selection import train_test_split
# 파이썬에는 PyPI라는 페키지 관리 시스템. 패키지를 다운로드하는 관리도구로 pip가 있음
# 명령프롬프트에서 pip install 패키지명 입력 후 설치

# from을 이용하여 time 모듈을 import하세요
from time import time
# now_time에 현재 시간을 대입
now_time = time()
print(now_time)     # 1591437386.345313
