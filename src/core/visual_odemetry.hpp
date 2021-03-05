#pragma once

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "feature_detector.hpp"

using namespace std;
using namespace cv;

class VisualOdemetry {
public:
	VisualOdemetry() {}
	~VisualOdemetry() {}

	void depth_calibration(cv::VideoCapture& camera);
	void initialize(Mat& img_initial_frame);
	void pose_estimation_pnp(Eigen::Matrix4f& T, VOFeatures& ref_frame_features,
                                 VOFeatures& curr_frame_features, vector<DMatch>& feature_matches);
	void estimate(cv::Mat& new_img);

private:
	VOFeatureDetector feature_detector;

	Mat last_frame_img;
	VOFeatures last_features;

	cv::Mat intrinsic_matrix;

	vector<DMatch> feature_matches;

	Eigen::Matrix4f T;

	/* debug visualization's data */
	Mat init_frame_img;
};
