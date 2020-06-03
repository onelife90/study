import numpy as np
import pandas as pd

#1-1. 데이터 읽어오기
samsung = pd.read_csv('./data/csv/samsung.csv', index_col=0, header=0, sep=',', encoding='CP949')
hite = pd.read_csv('./data/csv/hite.csv', index_col=0, header=0, sep=',', encoding='CP949')
# header=0, 헤더를 0행으로 지정. 0행==(시가, 고가, 저가, 종가, 거래량)
# (시가, 고가, 저가, 종가, 거래량)의 헤더가 없는 경우 : header=None
# CP949 한글 처리
# print(samsung)          # [700 rows x 1 columns]
# print(hite.head())
#                 시가      고가      저가      종가        거래량
# 일자
# 2020-06-02  39,000     NaN     NaN     NaN        NaN
# 2020-06-01  36,000  38,750  36,000  38,750  1,407,345
# 2020-05-29  35,900  36,750  35,900  36,000    576,566
# 2020-05-28  36,200  36,300  35,500  35,800    548,493
# 2020-05-27  35,900  36,450  35,800  36,400    373,464
# print(samsung.shape)    # (700, 1)
# print(hite.shape)       # (720, 5)

# Nan 제거 1
samsung = samsung.dropna(axis=0)
# dropna = nan을 제거 // axis=0(영행) df에서는 행이 우선
# print(samsung)           # [509 rows x 1 columns]
# print(samsung.shape)     # (509, 1)
hite = hite.fillna(method='bfill')
# print("hite_bfill: \n", hite)
# fillna = nan 값을 method(방법) bfill=back fill로 채운다

#                 시가      고가      저가      종가        거래량
# 일자
# 2020-06-02  39,000  38,750  36,000  38,750  1,407,345
# 2020-06-01  36,000  38,750  36,000  38,750  1,407,345
hite = hite.dropna(axis=0)
# print(hite)
# axis=0 nan이 있는 행이 다 drop

# Nan 제거 2
# hite = hite[0:509]
# 이렇게 슬라이싱을 하면 에러가 많이 발생
# hite.iloc[0, 1:5] = [10,20,30,40]
# hite.loc['2020-06-02', '고가':'거래량'] = ['100', '200', '300', '400']
# loc=location! 위치!
# loc 사용 시, str형식으로 쓸 때는 명확하게 컬럼명을 표시. 뒤의 컬럼-1이 아님을 주의!(숫자로 명시하는 iloc가 아니기 때문에)
# print(hite)

# 결측치 제거 다른 방법
# nan을 제거하지 않고, model fit(1:4열)하고 predict시, nan을 y_pred로 예측하여 넣는 방법도 있다
#                 시가      고가      저가      종가        거래량
# 일자
# 2020-06-02  39,000     NaN     NaN     NaN        NaN

#1-2. 오름차순으로 정렬
samsung = samsung.sort_values(['일자'], ascending=True)
hite = hite.sort_values(['일자'], ascending=True)
# print(samsung)
# print(hite)

#1-3. 수치형 데이터(int)로 변환
for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',', ''))
    # iloc = index location 모든 인덱스 행의 0번째 컬럼을 replace
# print(samsung)
# print(type(samsung.iloc[0,0]))      # <class 'int'>

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',', ''))
# print(len(hite.iloc[i]))=5
# i번째 인덱스의 개수? 5개(5컬럼)
# print(hite)
# print(type(hite.iloc[1,1]))         # <class 'int'>

# print(samsung.shape)     # (509, 1)
# print(hite.shape)        # (509, 5)

#1-4. numpy로 저장
samsung = samsung.values
hite = hite.values
# print(type(hite))       # <class 'numpy.ndarray'>

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)
