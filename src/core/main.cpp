#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "feature_detector.hpp"
#include "visual_odemetry.hpp"

#define IMAGE_WIDTH  640
#define IMAGE_HEIGHT 480

using namespace std;
using namespace cv;

int main()
{
	cv::VideoCapture camera(4, cv::CAP_V4L);
        camera.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
        camera.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);

	if(camera.isOpened() != true) {
		cout << "failed to open the camera.\n";
		return -1;
	}

	Mat raw_img;

	VisualOdemetry visual_odemetry;
	visual_odemetry.scale_calibration(camera);

	/* initialization */
	while(!camera.read(raw_img));
	visual_odemetry.initialize(raw_img);

	while(1) {
		while(!camera.read(raw_img));
		visual_odemetry.estimate(raw_img);
	}

	return 0;
}
