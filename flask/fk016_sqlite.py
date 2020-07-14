import sqlite3

#1-1. sqlite3에 연결 : test.db가 생성되지 않았으면 현재 폴더 하위에 자동으로 생성
conn = sqlite3.connect("test.db")

#1-2. connection 객체로부터 cursor() 메서드를 호출. DB 커서는 Fetch 동작을 관리하는데 사용
cursor = conn.cursor()

#1-3. cursor 실행
# sql로 들어가기(삽입, 삭제, 수정 가능)
# TABLE==DB에 쌓이는 csv 파일이라 생각하면 쉬움
cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT, FoodName TEXT, Company TEXT, Price INTEGER)""")
# table만 생성했기때문에 데이터가 없는 상태

#1-4. 중복된 데이터가 삭제되는 delete sql 커서 실행
sql = "DELETE FROM supermarket"
cursor.execute(sql)

#2. 데이터 넣기
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (1,'과일','자몽','마트',1500))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (2,'음료','망고주스','편의점',1000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (3,'고기','소고기','하나로마트',10000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (4,'박카스','약','약국',5000))

sql = "SELECT * FROM supermarket"
# sql = "SELECT Itemno, Category, FoodName, Company, Price FROM sypermarket"
cursor.execute(sql)

rows = cursor.fetchall()

#3. 데이터 넣어서 출력해보자
for row in rows:
    print(str(row[0])+" "+str(row[1])+" "+str(row[2])+" "+str(row[3])+" "+str(row[4]))

# 삽입, 삭제, 수정 시 commit 필요!!
conn.commit()

#4. 커서 끊기
conn.close()
