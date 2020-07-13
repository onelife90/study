from flask import Flask
app = Flask(__name__)

from flask import make_response

# cookie = 클라이언트의 PC에 텍스트 파일 형태로 저장되는 것으로 일반적으로 시간이 지나면 소멸
@app.route('/')
def index():
    # make_response() 함수는 사용자에게 반환할 뷰 함수를 생성한 후, 그대로 묶어두는 역할
    response = make_response('<h1> 잘 따라 치시오!!! </h1>')
    # set_cookie() 함수를 사용해 쿠키 생성. 쿠키 이름 'answer'
    response.set_cookie('answer', '42')
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

# 쿠키와 다르게 세션과 관련된 데이터는 서버에 저장
# 서버에서 관리할 수 있다는 점에서 안전성이 좋아 보통 로그인 관련으로 사용
