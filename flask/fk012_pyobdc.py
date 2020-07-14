# ODBC(Open Database Connectivity) : 마이크로소프트가 만든 DB에 접근하기 위한 소프트웨어
import pyodbc as pyo

#1-1. 변수명 지정
server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

#1-2. pyodbc에 connect
conn = pyo.connect("DRIVER={ODBC Driver 17 for SQL Server}; SERVER=" +server+ '; PORT=1433; DATABASE=' +database+
                   "; UID=" +username+"; PWD=" +password)
cursor = conn.cursor()

# SELECT*FROM== ~로부터 검색하여 가져오겠다
tsql = "SELECT * FROM iris2;"

#2. 커서 실행
# cursor.execute(tsql)==tsql에서 cursor를 실행하겠다. 이 cursor는 connect에 연결이 되어있음
# connection 객체로부터 cursor() 메서드를 호출. DB cursor는 Fetch 동작을 관리하는데 사용
with cursor.execute(tsql):
    row = cursor.fetchone()

    while row:
        print(str(row[0])+" "+str(row[1])+" "+str(row[2])+" "+str(row[3])+" "+str(row[4]))
        row = cursor.fetchone()

#3. 커서 끊기
conn.close()
