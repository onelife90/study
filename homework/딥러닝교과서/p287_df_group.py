# df의 특정 열에서 동일한 값의 행을 집계하는 것 : 그룹화
# df.groupby("컬럼") 지정한 컬럼을 그룹화
# 하지만 groupby 객체는 반환하나 그룹화된 결과를 표시 X
# 그룹화된 결과를 표시하려면? mean(), sum()등 통계함수 이용
import  pandas as pd
pre_df = pd.DataFrame([["강릉",1040, 213527, "강원도"],
                       ["광주", 430,1458915,"전라도"],
                       ["평창",1463,42218,"강원도"],
                       ["대전", 539,1476955,"충청도"],
                       ["단양", 780,29816,"충청도"]],
                    columns =["Prefecture", "Area", "Population", "Region"])
print(pre_df)
#   Prefecture  Area  Population Region
# 0         강릉  1040      213527    강원도
# 1         광주   430     1458915    전라도
# 2         평창  1463       42218    강원도
# 3         대전   539     1476955    충청도
# 4         단양   780       29816    충청도
# pre_df를 Region으로 그룹화하여 group_df에 대입
group_df = pre_df.groupby("Region")
# pre_df의 Area와 Population의 평균을 mean_df에 대입
mean_df = group_df.mean()
print(mean_df)
#           Area  Population
# Region
# 강원도     1251.5    127872.5
# 전라도      430.0   1458915.0
# 충청도      659.5    753385.5
