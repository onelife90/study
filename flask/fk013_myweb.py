# iris 데이터를 웹상으로 구현해보자
# ODBC(Open Database Connectivity) : 마이크로소프트가 만든 DB에 접근하기 위한 소프트웨어
import pyodbc as pyo

#1-1. 변수명 지정
server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

#1-2. pyodbc에 connect
# connection 객체로부터 cursor() 메서드를 호출. DB 커서는 Fetch 동작을 관리하는데 사용
conn = pyo.connect("DRIVER={ODBC Driver 17 for SQL Server}; SERVER=" +server+ '; PORT=1433; DATABASE=' +database+
                   "; UID=" +username+"; PWD=" +password)
cursor = conn.cursor()

# SELECT*FROM== ~로부터 검색하여 가져오겠다
tsql = "SELECT * FROM iris2;"

'''
with cursor.execute(tsql):
    row = cursor.fetchone()

    while row:
        print(str(row[0])+" "+str(row[1])+" "+str(row[2])+" "+str(row[3])+" "+str(row[4]))
        row = cursor.fetchone()
'''

#2. flask와 연동되는 코딩
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/sqltable")
def showsql():
    cursor.execute(tsql)
    return render_template("myweb.html", rows=cursor.fetchall())
    # myweb.html에서 모든 컬럼과 행을 cursor로 가져오겠다. cursor는 tsql에서 실행

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

conn.close()
