#include <opencv2/opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;


int main()
{
	Mat img_color;

	//비디오 캡쳐 초기화
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "에러 - 카메라를 열 수 없습니다.\n";
		return -1;
	}


	while (1)
	{
		// 카메라로부터 캡쳐한 영상을 frame에 저장합니다.
		cap.read(img_color);
		if (img_color.empty()) { 
			cerr << "빈 영상이 캡쳐되었습니다.\n";
			break;
		}

		// 영상을 화면에 보여줍니다. 
		imshow("Color", img_color);


		// ESC 키를 입력하면 루프가 종료됩니다. 

		if (waitKey(25) >= 0)
			break;
	}


	return 0;
}
