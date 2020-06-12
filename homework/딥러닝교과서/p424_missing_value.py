# df 구조에서 결측치 제거
# 데이터가 누락된행(nan을 가진 행)을 통째로 지우는 것을 리스트와이즈 삭제 listwise deletion
# 어떻게 사용? dropna()메서드를 사용하면 nan이 있는 모든 행이 삭제
# 사용 가능한 데이터만 활용하는 방법도 있음. 결손이 적은열만 남기는 것을 페어와이즈 삭제 pairwise deletion
import numpy as np
from numpy import nan as NA
import pandas as pd

np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10,4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA

solve = sample_data_frame[[0,2]].dropna()
print(solve)
# dropna 실행 후 sample_data_frame를 출력하니 원본 데이터 출력
# 새로운 변수명으로 출력물 확인
