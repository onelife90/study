from flask import Flask

app = Flask(__name__)

# /<name> --> def(name) --> return %name
# name이 연결되어 찾아감
# 즉, url 규칙을 받아 해당하는 규칙의 url로 요청이 들어온 경우 등록한 함수를 실행해라
@app.route("/<name>")
def user(name):
    return '<h1>hello, %s !!!</h1>' %name

@app.route("/user/<name>")
def user2(name):
    return '<h1>hello, user/ %s !!!</h1>' %name

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
