#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "feature_detector.hpp"
#include "visual_odemetry.hpp"

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

	Mat raw_img;

	/* initialization */
	while(!cap.read(camera));
	raw_img = cv::Mat(camera);

	VOFeatureDetector feature_detector;
	VisualOdemetry visual_odemetry(raw_img);

	while(cap.read(camera)) {
		raw_img = cv::Mat(camera);
		visual_odemetry.estimate(raw_img);
	}

	return 0;
}
