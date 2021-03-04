#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class VOFeatureDetector {
public:
	void match(Mat& img_last, Mat& img_nowi);

private:
	/* debug images */
	Mat keypoint_img_last, keypoint_img_now;
	Mat feature_match_img, filtered_feature_match_img;
};
