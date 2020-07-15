from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

#1. 데이터베이스 연결 후 cursor 실행
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")
print(cursor.fetchall())


#2. wanggun.db에서 모든 행을 땡겨오는 라우트 발동
@app.route('/')
def run():
    conn = sqlite3.connect("./data/wanggun.db")
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall()
    return render_template("board_index.html", rows=rows)


#3-1. 웹상으로 수정할 수 있는 게시판을 만들자
# /modi로 보내는 라우트가 필요
@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general WHERE id = "+str(id))
    rows = c.fetchall();
    return render_template("board_modi.html", rows=rows)


#3-2. /addred로 라우트를 발동하여 웹상에서 수정하는 app 만들기
# POST==전송되는 http 내부에 데이터를 추가하여 보내는 방식
@app.route('/addred', methods=["POST", "GET"])
def addrec():
    if request.method == 'POST':
        print(request.form['war'])
        print(request.form['id'])
        
        # 파이썬에서 예외처리
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/wanggun.db") as conn:
                cur = conn.cursor()
                # 수정을 실행하는 명령어 UPDATE
                cur.execute("UPDATE general SET war="+str(war)+" WHERE id= "+str(id))
                # 수정을 행하기 때문에 commit 필수
                conn.commit()
                msg = "정상적으로 입력되었습니다"
        except:
            # 에러가 발생하게 되면 돌려라
            conn.rollback()
            msg = "입력과정에서 에러가 발생했습니다"

        finally:
            return render_template("board_result.html", msg=msg)
            conn.close()

app.run(host='127.0.0.1', port=5001, debug=False)
