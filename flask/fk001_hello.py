# ping-pong 내 걸 치고 상대방이 받는 것
# [152.209.222.142] naver.com
# 고정 ip

# ipconfig(내 ip 확인)
# 플라스크 : 파이썬으로 작성된 마이크로 웹 프레임워크

from flask import Flask
app = Flask(__name__)

# /==host를 입력하고 /를 입력하면 return을 반환해주는 함수
# <h1>==큰 글씨
@app.route('/')
def hello333():
    return "<h1>hello AI world</h1>"

@app.route('/bit')
def hello():
    return "<h1>hello bit computer world</h1>"

@app.route('/bit/bitcamp')
def hello_bit():
    return "<h1>hello bitcamp world</h1>"

@app.route('/gemma')
def hello_gemma():
    return "<h1>hello GEMMA world\n puhahahahah </h1>"
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)
    # '127.0.0.1' == 내 자신
