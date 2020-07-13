from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello123():
    return "<h1>hello world</h1>"

# ping을 주면 pong을 get해라
# 즉, url 규칙을 받아 해당하는 규칙의 url로 요청이 들어온 경우 등록한 함수를 실행해라
@app.route('/ping', methods=['GET'])
def ping():
    return "<h1>pong</h1>"

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5555, debug=False)
