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

	Eigen::MatrixXf A(8, 9);

	int r = 0;
	for(cv::DMatch m:feature_matches) {
		double u1, v1, u2, v2;
		u1 = last_features.keypoints[m.trainIdx].pt.x;
		v1 = last_features.keypoints[m.trainIdx].pt.y;
		u2 = new_features.keypoints[m.trainIdx].pt.x;
		v2 = new_features.keypoints[m.trainIdx].pt.y;
		printf("(%lf, %lf) -> (%lf, %lf)\n", u1, v1, u2, v2);

		if(r < 8) {
			A(r, 0) = u1*u1;
			A(r, 1) = u1*v2;
			A(r, 2) = u1;
			A(r, 3) = v1*u2;
			A(r, 4) = v1*v2;
			A(r, 5) = v1;
			A(r, 6) = u2;
			A(r, 7) = v2;
			A(r, 8) = 1;
			r++;
		}
	}

	cout << A << endl;

	imshow("img1", test_img1);
	imshow("img2", test_img2);

#endif
	Eigen::JacobiSVD<MatrixXf> svd(A.transpose() * A, ComputeThinU | ComputeThinV);

	cout << "SVD of AtA" << endl;
	cout << "singular values" << endl << svd.singularValues() << endl;
	cout << "U:" << endl << svd.matrixU() << endl;
	cout << "V:" << endl << svd.matrixV() << endl;

	/* find smallest singular value */
	double min = svd.singularValues()[0];
	int min_index = 0;
	for(int i = 1; i < 9; i++) {
		if(svd.singularValues()[i] < min) {
			min = svd.singularValues()[i];
			min_index = i;
		}
	}
	printf("index %d element has the smallest singular value\n", min_index);

	/* restore R and t from E */
	

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
