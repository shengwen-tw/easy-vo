#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

int main()
{
	cv::Mat frame;
	cv::VideoCapture cap(0, cv::CAP_V4L);

	if(cap.isOpened() != true) {
		return -1;
	}

	while(cap.read(frame)) {
		cv::Mat src = cv::Mat(frame);
		cv::imshow( "window",  src );
		cv::waitKey(30);
	}

	return 0;
}
