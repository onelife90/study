# 웹캠으로 들어오는 영상을 흑백 영상으로 변환하여 화면에 디스플레이해보자

import numpy as np
import cv2

def showvideo():
    try:
        print("카메라를 구동합니다")
        cap = cv2.VideoCapture(0)
        # 비디오 캡쳐를 위한 VideoCapture 객체 생성. 인자는 장치 인덱스 또는 비디오 파일 이름 지정
        # 캠이 하나만 부착되어 있으면 0을 지정
    except:
        print("카메라 구동 실패")
        return

    cap.set(3,480)
    cap.set(4,320)

    # 라이브로 들어오는 비디오를 프레임별로 캡쳐하고 이를 화면에 디스플레이. 특정 키를 누를 때까지 무한루프
    while True:
        ret, frame = cap.read()
        # cap.read()==비디오의 한 프레임씩 읽기
        if not ret:
            print("비디오 읽기 오류")
        # 흑백 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("video", gray)

        k = cv2.waitKey(1)& 0xFF
        if k == 27:
            break
    
    # 오픈한 cap 객체를 반드시 해제
    cap.release()
    # 생성한 모든 윈도우 제거
    cv2.destroyAllWindows()

showvideo()
        