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
	VisualOdemetry(Mat& img_initial_frame);
	void pose_estimation_pnp(Eigen::Matrix4f& T, VOFeatures& ref_frame_features,
                                 vector<cv::Point3f>& reference_points_3d,
                                 VOFeatures& curr_frame_features, vector<DMatch>& feature_matches);
	void estimate(cv::Mat new_img);

private:
	VOFeatureDetector feature_detector;

	VOFeatures ref_frame_features;
	vector<cv::Point3f> reference_keypoints_3d;

	cv::Mat intrinsic_matrix;

	vector<DMatch> feature_matches;

	/* debug visualization's data */
	Mat init_frame_img;
};
