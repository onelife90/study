from flask import Flask, Response, make_response

app = Flask(__name__)

@app.route('/')
def response_test():
    custom_response = Response("Custom Response", 200, {"Program" : "Flask Web Application"})
    print("[★]Custom Response")
    return make_response(custom_response)

# app.route의 친구들
@app.before_first_request
def before_first_request():
    print("[1]앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다")
    # 웹서버를 실행시키면, 라우트가 실행되기 전에 before_first_request가 실행
    # 바로 응답하지 않으려는 경우에 사용

@app.before_request
def before_request():
    print("[2]매 HTTP 요청이 처리되기 전에 실행됩니다")
    # 웹서버를 실행시키면, 라우트가 실행되기 전에 before_request가 실행 --> app.route

@app.after_request
def after_request(response):
    print("[3]매 HTTP 요청이 처리되고 나서 실행됩니다")
    return response
    # after_request는 함수에 return을 해줘야 함

@app.teardown_request
def teardown_request(exception):
    print("[4]매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다")

@app.teardown_appcontext
def teardown_appcontext(exception):
    print("[5]HTTP 요청의 어플리케이션 컨텍스트가 종료될 때 실행된다")
    # flask의 설계에 숨겨진 사상은 코드가 수행되는 두가지 다른 상태가 있다는 것
    # 컨텍스트는 요청당 하나씩 생성되거나 직접 사용하는 경우 캐시 리소스에 사용

if __name__ == '__main__':
    app.run(host='127.0.0.1')
    # flask의 default port=5000


# [1]앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다             # before
# [2]매 HTTP 요청이 처리되기 전에 실행됩니다                        # before
# [★]Custom Response
# [3]매 HTTP 요청이 처리되고 나서 실행됩니다                        # after
# [4]매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다        # after
# [5]HTTP 요청의 어플리케이션 컨텍스트가 종료될 때 실행된다         # context         

# 127.0.0.1 - - [13/Jul/2020 20:23:08] "mGET / HTTP/1.1" 200 -
# [2]매 HTTP 요청이 처리되기 전에 실행됩니다                        # before
# [★]Custom Response           
# [3]매 HTTP 요청이 처리되고 나서 실행됩니다                        # after
# [4]매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다        # after
# [5]HTTP 요청의 어플리케이션 컨텍스트가 종료될 때 실행된다         # context
