# 플라스크 : 파이썬으로 작성된 마이크로 웹 프레임워크(백엔드)
# html : 웹페이지(프론트엔드)

import os
from flask import Flask, url_for, render_template

app = Flask(__name__, static_folder='static')

# /==host를 입력하고 /를 입력하면 return을 반환해주는 함수
# <h1>==큰 글씨
@app.route('/')
def videolist():
    return render_template('videolist.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)
    # '127.0.0.1' == 내 자신