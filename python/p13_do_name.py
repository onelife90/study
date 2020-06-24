# 함수 불러오기
import p11_car_module
# 함수 안에 프린트문이 있으면 그거까지 싹 다 가져와서 import
import p12_tv_module

print("=============")
print("do.py의 module 이름은 ", __name__)
print("=============")

p11_car_module.drive()
p12_tv_module.watch()

# 운전하다
# car.py의 module 이름은  p11_car_module
# 시청하다
# tv.py의 module 이름은  p12_tv_module
# =============
# do.py의 module 이름은  __main__
# =============
# 운전하다
# 시청하다


# 자기 자신에서 __name__은 __main__
# import하면 __name__은 파일명
