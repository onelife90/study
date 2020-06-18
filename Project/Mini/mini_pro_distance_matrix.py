import pandas as pd
import json
import googlemaps
from itertools import tee

#1. 서초동 주소를 좌표로 변환한 csv 파일 읽어오기
df = pd.read_csv('./mini/seocho_addr.csv', sep=',', encoding="UTF8", header=0, index_col=None)

# 구글 API키를 저장한 json 파일 직렬 "r" 형태로 파이썬에 불러오기
with open("./mini/google_keys.json","r") as clientJson:
    client = json.load(clientJson)
    gmaps = googlemaps.Client(key=client["key"])
    
# df 내부 2003개의 행을 (위도, 경도)로 묶어주는 함수
def pairwise(iterable):
    x,y = tee(iterable)
    next(y,None)
    return zip(x,y)

# 각 위치마다 목적지까지의 거리 계산 값을 넣어줄 리스트 형성
list = [0]

# for문으로 2003개의 (위도, 경도) 형성
for (i1,row1), (i2,row2) in pairwise(df.iterrows()):
    
    latorigin = row1['Latitude']
    longorigin = row1['Longitude']
    origins = (latorigin,longorigin)
    
    latdest = row2['Latitude']
    longdest = row2['Longitude']
    destination = (latdest,longdest)

    result = gmaps.distance_matrix(origins, destination, mode='transit')["rows"][0]["elements"][0]["distance"]["value"]
    # 0번째 인덱스 행에 0번째 요소는 distance이고 계산 값을 넣겠다
    print(result)

    list.append(result)

# 리스트 형태인 Distance를 df에 컬럼 추가
df['Distance'] = list

# csv파일 저장하기
df.to_csv('./mini/seoch_distances.csv', sep=';', index=None, header= ['Id','Addr','Latitude','Longitude','Distance'])
