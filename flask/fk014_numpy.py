# 넘파이로 저장도 가능하다
import pymssql as ms
import numpy as np

#1. 데이터베이스에 있는 db 연결하기
conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitdb')

#1-1. connection 객체로부터 cursor() 메서드를 호출. DB 커서는 Fetch 동작을 관리하는데 사용
cursor = conn.cursor()

#1-2. cursor를 실행. SELECT*FROM== ~로부터 검색하여 가져온다
cursor.execute("SELECT * FROM iris2;")
row = cursor.fetchall()
print(f"row: {row}")
# 개행없이 쭉 나옴

asarray_row = np.asarray(row)
print(f"asarray_row: {asarray_row}")
# 배열형태로 출력
print(f"asarray_row.shape: {asarray_row.shape}")    # asarray_row.shape: (150, 5)
print(type(asarray_row))    # <class 'numpy.ndarray'>

#2. 넘파이 저장
np.save('./data/test_flask_iris2.npy', asarray_row)

#3. 커서 끊기
conn.close()
