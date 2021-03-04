#pragma once

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

class VisualOdemetry {
public:
	VisualOdemetry();
	void pose_estimation_pnp(Eigen::Matrix4f& T, vector<cv::Point3f>& reference_points_3d,
                                 vector<cv::KeyPoint>& feature_keypoints_2d, vector<DMatch>& feature_matches);
	void estimate(cv::Mat new_image);

private:
	cv::Mat intrinsic_matrix;
};
