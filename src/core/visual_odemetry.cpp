#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "visual_odemetry.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

bool calib_exit_signal = false;

void VisualOdemetry::initialize(Mat& img_initial_frame)
{
	/* intrinsic matrix */
	double fx, fy, cx, cy;
	fx = 656.24987;
	fy = 656.10660;
	cx = 327.36105;
	cy = 240.03464;

	intrinsic_matrix = (cv::Mat_<double>(3,3) << fx,  0, cx,
	                                              0, fy, cy,
	                                              0,  0,  1);

	/* exract features from initial frame */
	feature_detector.extract(img_initial_frame, last_features);

	/* set intial frame's relative position to be zero */
	for(int i = 0; i < last_features.keypoints.size(); i++) {
		last_features.points_3d.push_back(cv::Point3f(0, 0, 0));
	}

	/* debug */
	img_initial_frame.copyTo(last_frame_img);
}

void VisualOdemetry::estimate_non_scaled_essential_matrix()
{
#if 1
	/* test code */
	Mat test_img1 = imread("/home/shengwen/test_data/1.png");
	Mat test_img2 = imread("/home/shengwen/test_data/2.png");

	imshow("img1", test_img1);
	imshow("img2", test_img2);

	VOFeatures last_features, new_features;
	feature_detector.extract(test_img1, last_features);
	feature_detector.extract(test_img2, new_features);

		vector<DMatch> feature_matches;
	feature_detector.match(feature_matches, last_features, new_features);

	feature_detector.plot_matched_features(test_img1, test_img2, last_features, new_features, feature_matches);

	imshow("img1", test_img1);
	imshow("img2", test_img2);

	while(1) {cv::waitKey(30);}
#endif
	float u1[8], v1[8], u2[8], v2[8];

	/* eight point method */
	Eigen::MatrixXf A(8, 9);
	A << u1[0]*u1[0], u1[0]*v2[0], u1[0], v1[0]*u2[0], v1[0]*v2[0], v1[0], u2[0], v2, 1,
             u1[1]*u1[1], u1[1]*v2[1], u1[1], v1[1]*u2[1], v1[1]*v2[1], v1[1], u2[1], v2, 1,
             u1[2]*u1[2], u1[2]*v2[2], u1[2], v1[2]*u2[2], v1[2]*v2[2], v1[2], u2[2], v2, 1,
             u1[3]*u1[3], u1[3]*v2[3], u1[3], v1[3]*u2[3], v1[3]*v2[3], v1[3], u2[3], v2, 1,
             u1[4]*u1[4], u1[4]*v2[4], u1[4], v1[4]*u2[4], v1[4]*v2[4], v1[4], u2[4], v2, 1,
             u1[5]*u1[5], u1[5]*v2[5], u1[5], v1[5]*u2[5], v1[5]*v2[5], v1[5], u2[5], v2, 1,
             u1[6]*u1[6], u1[6]*v2[6], u1[6], v1[6]*u2[6], v1[6]*v2[6], v1[6], u2[6], v2, 1,
             u1[7]*u1[7], u1[7]*v2[7], u1[7], v1[7]*u2[7], v1[7]*v2[7], v1[7], u2[7], v2, 1;

	Eigen::MatrixXf AtA(9, 9);
	AtA = A * A.transpose();

	Eigen::JacobiSVD<MatrixXf> svd(AtA, ComputeThinU | ComputeThinV);

	cout << "SVD of AtA:";
	cout << "singular values" << endl << svd.singularValues() << endl;
	cout << "U:" << endl << svd.matrixU() << endl;
	cout << "V:" << endl << svd.matrixV() << endl;

	while(1) {cv::waitKey(30);}
}

void VisualOdemetry::depth_calibration(cv::VideoCapture& camera)
{
	cv::Mat raw_img;

	auto imshow_callback = [](int event, int x, int y, int flags, void* param) {
	        if(event == CV_EVENT_LBUTTONDOWN) {
			calib_exit_signal = true;
	        }
	};
	namedWindow("scale calibration");
	setMouseCallback("scale calibration", imshow_callback, NULL);

	printf("click the window to collect the calibration frame 1\n");
	calib_exit_signal = false;
	while(!calib_exit_signal) {
		while(!camera.read(raw_img));
		imshow("scale calibration", raw_img);
		cv::waitKey(30);
	}

	printf("click the window to collect the calibration frame 2\n");
	calib_exit_signal = false;
	while(!calib_exit_signal) {
		while(!camera.read(raw_img));
		imshow("scale calibration", raw_img);
		cv::waitKey(30);
	}

	/* calculate scaling factor via least square method */
}

void VisualOdemetry::pose_estimation_pnp(Eigen::Matrix4f& T,
                VOFeatures& ref_frame_features,
                VOFeatures& curr_frame_features,
                vector<DMatch>& feature_matches)
{
	vector<Point3f> obj_points_3d;
	vector<Point2f> img_points_2d;

	for(cv::DMatch m:feature_matches) {
		obj_points_3d.push_back(ref_frame_features.points_3d[m.queryIdx]);
		img_points_2d.push_back(curr_frame_features.keypoints[m.trainIdx].pt);
	}

	Mat rvec, tvec, inliers;
	solvePnPRansac(obj_points_3d, img_points_2d, intrinsic_matrix, Mat(), rvec, tvec,
	               false, 100, 4.0, 0.99, inliers);

	int inlier_cnt = inliers.rows;
	printf("pnp inliers count = %d\n", inlier_cnt);

	T << rvec.at<float>(0, 0), rvec.at<float>(0, 1), rvec.at<float>(0, 2), tvec.at<float>(0, 0),
             rvec.at<float>(1, 0), rvec.at<float>(1, 1), rvec.at<float>(1, 2), tvec.at<float>(1, 0),
             rvec.at<float>(2, 0), rvec.at<float>(2, 1), rvec.at<float>(2, 2), tvec.at<float>(2, 0),
                                0,                    0,                    0,                    1;
}

void VisualOdemetry::estimate(cv::Mat& new_img)
{
	Eigen::Matrix4f T_last_to_now;

	VOFeatures new_features;
	feature_detector.extract(new_img, new_features);

	feature_detector.match(feature_matches, last_features, new_features);

	pose_estimation_pnp(T_last_to_now, last_features, new_features, feature_matches);

	feature_detector.plot_matched_features(last_frame_img, new_img, last_features, new_features, feature_matches);

	//new_img.copyTo(last_frame_img);
	//last_features = new_features;

	//printf("estimated position = %lf, %lf, %lf\n", T(0, 3), T(1, 3), T(2, 3));
}
