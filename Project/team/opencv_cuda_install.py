'''openCV 4.2.0 with CUDA 설치 과정

[준비사항]
1. Nvidia 그래픽 카드 드라이버 최신 업데이트
2. Visual Studio 2017 설치

[설치순서]
1. cuda toolkit 설치 (10.0)
2. cudnn 설치 (7.6.2)
- CUDA® Deep Neural Network library
3. cmake 설치 (3.16.2)
- 오픈소스를 다운로드 받아서 빌드를 하기 위해 필요한 솔루션 파일 또는 메이크 파일을 만들어주는 기능
4. python 설치 (3.7.6) / tensorflow-gpu 설치
5. openvino 설치 (2019_R3.1)
- 딥러닝 가속화 라이브러리
6. openCV 4.2.0 소스 코드 다운로드 (main & extra)
- C:\opencv\build 경로에 cmake 프로그램을 이용하여 build할 솔루션 파일과 프로젝트 파일들 저장
- CMakeLists.txt에 있는 코드를 분석하여 자동으로 build
- build_examples : opencv에서 제공하는 예제 사용가능
- build_opencv_world : opencv_world 라는 하나의 통합 모듈을 사용
- opencv_extra_module_path : C:\opencv\opencv_contrib-4.2.0\modules
- with_cuda
- opencv_dnn_cuda
- with_inf_engine : openvino를 사용할 거면 체크
- 다시 한번 configure
- CUDA_ARCH_BIN 에러 발생 -> wiki CUDA 구글링하여 버전 표 확인 -> 설치된 그래픽 카드 RTX 2080 == 7.5ver
- inferenceEngine_DIR : path가 제대로 설정되어있어야함 C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\share
7. generate : cmake 설정 -> opencv.sln 파일 생성
8. visual studio 2017에서 빌드
- visual studio version seletor 선택하면 자동으로 버전확인하여 앱 선택
'''
