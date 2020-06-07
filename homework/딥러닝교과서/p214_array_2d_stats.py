# 통계함수 : ndarray배열 전체 또는특정 축을 중심으로 수학적 처리를 수행하는 함수, 메서드
# 배열 요소의 평균 반환 mean(), np.average(), 최대값/최소값 반환 np.max(), np.min(), 요소의 최대값/최소값 np.argmax(), np.argmin()
# 표준 편차와 분산을 반환 np.std(), np.var() 데이터의 편차를 나타내는 지표
import numpy as np
arr = np.arange(15).reshape(3,5)
# arr의 각 열의 평균 출력
print(arr.mean(axis=0))     # [5. 6. 7. 8. 9.]
# arr의 행 합계 출력
print(arr.sum(axis=1))      # [10 35 60]
# arr의 최소값 출력
print(arr.min())            # 0
# arr의 각 열의 최대값의 인덱스 번호 출력
print(arr.argmax(axis=0))   # [2 2 2 2 2]
