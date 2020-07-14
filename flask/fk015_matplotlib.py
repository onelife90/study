# 웹상에서 그림을 그려보자
from flask import Flask, render_template, send_file, make_response
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

#1. 플라스크에 가동할 앱준비
app = Flask(__name__)

#2. mypic.html에 입력했던 이미지가 웹으로 출력되는 route 정의
@app.route("/mypic")
def mypic():
    return render_template("mypic.html")

#3. 이미지를 웹상으로 바로 보여줄 route 정의
@app.route("/plot")
def plot():
    fig, axis = plt.subplots(1)

    #3-1. 데이터 준비
    x = [1,2,3,4,5]
    y = [0,2,1,3,4]
    
    #3-2. 데이터를 캔버스에 그린다
    axis.plot(x,y)
    canvas = FigureCanvas(fig)

    img = BytesIO()
    # BytesIO : 바이트 배열을 이진 파일로 다룰 수 있게 해주는 클래스
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype="image/png")

#4. 실행
if __name__ == '__main__':
    port = 5050
    app.debug = False
    app.run(port=port)
