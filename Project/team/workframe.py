''' 개념 및 용어 정리
openCV(Open Source Computer Vision Library)
- 조금이라도 영상처리가 들어간다면 필수적으로 사용하게 되는 라이브러리
- 프로그래밍 언어 : C / C++

python은 스크립트 언어이기 때문에 C / C++와 같은 컴파일 언어에 비해 속도가 느린 단점
그래서 우리는 성능적 이슈가 있는 부분을 C / C++로 구현한 후 파이썬으로 불러 사용하도록 파이썬 래퍼 생성

일단, OpenCV 4.2.0부터 DNN 실행 시 CUDA 백앤드 사용을 지원하기에
cuda를 사용하여 openCV를 빌드하자


<사용할 API>
darknet : C로 작성한 신경망 오픈소스
darkflow : C로 작성된 것이 아닌 tensorflow로 작성된  yolo
yolo : 기본적으로 제공되는 사전 훈련된 모델
cmake(3.16.2) : 오픈소스를 다운로드 받아서 빌드를 하기 위해 필요한 솔루션 파일 또는 메이크 파일을 만들어주는 기능


<필요사항>
1. 웹캠을 이용해 실시간 객체 추적을 하고싶다?
- CUDA, openCV로 darknet을 compile

2. yolo로 데이터 훈련해보고싶다?
- pascal VOC(Visual Object Classes) data

2-1. 그렇다면 라벨링 된 파일은?
- darknet wants a .txt file

3. openCV에 동영상 파일을 넣어서 실행한다면?
- ./darknet detector demo cfg/coco.data cfg/yolov2.cfg yolov2.weights <video file>
'''
