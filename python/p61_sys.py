# 시스템 import
import sys
print(sys.path)

from test_import import p62_import
p62_import.sum2()
# 이 import는 아나콘다 폴더에 들어있다
# 작업그룹 임포트 썸탄다

from test_import.p62_import import sum2
sum2()
# 작업그룹 임포트 썸탄다
