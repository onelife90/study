# 인터프리터는 고급 언어로 작성된 프로그램을 한 줄 단위로 받아들여 번역하고, 번역과 동시에 프로그램을 한 줄 단위로 즉시 실행시키는 프로그램
# 파이썬은 정의들을 파일에 넣고 스크립트나 인터프리터의 대화형 모드에서 사용할 수 있는 방법을 제공
# 이런 파일을 module
# module로부터 정의들이 다른 module이나 main module로 import될 수 있음
def drive():
    print("운전하다")

drive()

print("car.py의 module 이름은 ", __name__)
# __name__ 이 파일의 main 파일
# main module은 최상위 수준에서 액세스하는 변수들의 컬렉션
# module의 이름은 전역 변수 __name__으로 제공
