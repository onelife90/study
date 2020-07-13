from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)
    # 렌더링할 html 파일명, 이후에 html 파일에 전달할 변수를 넘겨준다
    # url을 통해 전달받은 값<name> --> render_template 안 name으로 전달
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
