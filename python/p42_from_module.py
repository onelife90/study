# machine이라는 모듈 car에서 drive라는 함수 패키지 import
# machine이라는 모듈 tv에서 watch라는 함수 패키지 import

from machine.car_drive_hamsu import drive
from machine.tv_watch_hamsu import watch

drive()
watch()

# 운전하다
# 시청하다

# machine이라는 모듈에서 car(모든 것이 담긴) import
# machine이라는 모듈에서 tv(모든 것이 담긴) import
from machine import car_drive_hamsu
from machine import tv_watch_hamsu

car_drive_hamsu.drive()
tv_watch_hamsu.watch()

# 운전하다
# 시청하다
