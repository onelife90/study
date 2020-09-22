from darknet import darknet
import cv2, numpy as np
# 파이썬에서 dll 파일 로딩하여 제공하는 함수 호출 가능하게 하는 모듈
from ctypes import *

#1. 사용할 변수 선언
config_path ="C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/train/my_yolov3.cfg"
weigh_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/train/my_yolov3_final.weights"
meta_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/train/my_data.data"
video_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/video/16-3.mp4"
threshold = 0.25

#2. 관련 파일 load
# cfg(모델관련)와 weight 파일 아스키코드로 load
net = darknet.load_net(bytes(config_path, "ascii"), bytes(weigh_path, "ascii"), 0) 
# data(classes, train, valid, names, backup의 경로가 명시된 텍스트파일)  아스키코드로 load
meta = darknet.load_meta(bytes(meta_path, "ascii"))
# 비디오의 프레임을 추출하는 VideoCapture 함수
cap = cv2.VideoCapture(video_path) 

#3. opencv(darknet)
i = 0
# video capture object open
while(cap.isOpened()):
    i += 1
    # read per video 1 frame
    # ret = True, image = read frame
    ret, image = cap.read()
    print(f'순번: {i}')
    # resize to between pixel values. INTER_AREA = pixel interpolation
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    if not ret: 
        break 
    
    frame = darknet.nparray_to_image(image)
    r = darknet.detect_image(net, meta, frame) 
    print(f'detect_img: {r}')

    boxes = [] 
 
    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 
        boxes.append((x, y, w, h))
 
    for k in range(len(boxes)): 
        x, y, w, h = boxes[k] 
        top = max(0, np.floor(x + 0.5).astype(int)) 
        left = max(0, np.floor(y + 0.5).astype(int)) 
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2) 
        cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)
 
    cv2.imshow('frame', image) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # free_image=C++ image processing library
    # multi-thread 32 bit support
    darknet.free_image(frame)

cap.release()