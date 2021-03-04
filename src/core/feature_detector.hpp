#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

typedef struct {
	std::vector<KeyPoint> keypoints;
	Mat descriptors;
} VOFeatures;

class VOFeatureDetector {
public:
	VOFeatureDetector();
	void extract(Mat& img, VOFeatures& features);
	void match(vector<DMatch>& feature_matches, VOFeatures& features1, VOFeatures& features2);
	void plot_matched_features(Mat& img1, Mat& img2,
		                   VOFeatures& features1, VOFeatures& features2,
                                   vector<DMatch>& feature_matches);

private:
	Ptr<ORB> orb;
};
