# redirect 다시 전달

from flask import Flask
from flask import redirect

# flask를 __name__으로 구성
app = Flask(__name__)

# route는 뒤에 나오는 함수와 항상 연결
@app.route('/')
def index():
    return redirect('http://www.naver.com')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7777, debug=False)
