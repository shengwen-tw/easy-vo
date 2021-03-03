#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "feature_detector.hpp"

using namespace std;
using namespace cv;

int main()
{
	cv::Mat camera;
	cv::VideoCapture cap(0, cv::CAP_V4L);

	if(cap.isOpened() != true) {
		cout << "failed to open the camera.\n";
		return -1;
	}

	Mat img_last, img_now;

	/* initialization */
	while(!cap.read(camera));
	img_last = cv::Mat(camera);

	VOFeatureDetector feature_detector;

	while(cap.read(camera)) {
		img_now = cv::Mat(camera);

		feature_detector.match(img_last, img_now);

		img_last = img_now;
	}

	return 0;
}
