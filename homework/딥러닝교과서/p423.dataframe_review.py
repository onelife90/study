# 문제
# attri_data_frame1에 attri_data_frame2 행을 추가해서 출력. 행을 추가시 ID의 오름차순 정렬, 행 번호도 오름차순

import pandas as pd
from pandas import Series, DataFrame

attri_data1 = {"ID": ["100","101","102","103","104","106","108","110","111","113"],
               "city": ["서울","부산","대전","광주","서울","서울","부산","대전","광주","서울"],
               "birth_year": [1990,1989,1992,1997,1982,1991,1988,1990,1995,1981],
               "name": ["영이","순돌","짱구","태양","션","유리","현아","태식","민수","호식"]} 
attri_data_frame1 = DataFrame(attri_data1) 
attri_data2 = {"ID": ["107","109"],
               "city": ["봉화", "전주"],
               "birth_year":[1994,1988]} 
attri_data_frame2 = DataFrame(attri_data2)  

df = attri_data_frame1.append(attri_data_frame2).sort_values(by="ID", ascending=True)
print(df)
