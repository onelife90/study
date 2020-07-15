from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello123():
    return "<h1>hello world</h1>"

# ping을 주면 pong을 get해라
# 'GET' == url에 파라미터를 함께 넘기는 방식
@app.route('/ping', methods=['GET'])
def ping():
    return "<h1>pong</h1>"

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5555, debug=False)
