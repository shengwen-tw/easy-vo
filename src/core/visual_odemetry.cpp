#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "visual_odemetry.hpp"
#include "se3_math.hpp"

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

void VisualOdemetry::estimate_essential_matrix(VOFeatures& ref_frame_features,
                                               VOFeatures& curr_frame_features,
                                               vector<DMatch>& feature_matches)
{
#if 0
	Mat test_img1 = imread("/home/shengwen/test_data/1.png");
	Mat test_img2 = imread("/home/shengwen/test_data/2.png");

	imshow("img1", test_img1);
	imshow("img2", test_img2);

	VOFeatures ref_frame_features, curr_frame_features,;
	feature_detector.extract(test_img1, ref_frame_features);
	feature_detector.extract(test_img2, curr_frame_features);

	vector<DMatch> feature_matches;
	feature_detector.match(feature_matches, ref_frame_features, curr_frame_features);

	feature_detector.plot_matched_features(test_img1, test_img2, ref_frame_features,
                                               curr_frame_features, feature_matches);
#endif
	vector<Point2f> points1, points2;
	for(int i = 0; i < feature_matches.size(); i++) {
		points1.push_back(ref_frame_features.keypoints[feature_matches[i].queryIdx].pt);
		points2.push_back(curr_frame_features.keypoints[feature_matches[i].trainIdx].pt);
	}

	Point2d principle_point(327.36105, 240.03464);
	int focal_length = 656.24987;

	/* solve essential matrix */
	Mat E = findEssentialMat(points1, points2, focal_length, principle_point, RANSAC);
	cout << "E:\n" << E << endl;

	/* factor out rotation and translation from essential matrix */
	Mat R, t;
	recoverPose(E, points1, points2, R, t, focal_length, principle_point);
	cout << "R:\n" << R << endl;
	cout << "t:\n" << t << endl;

	double  _sqrt = sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
	double roll = 57.2958 * atan2(R.at<double>(2,1) , R.at<double>(2,2));
	double pitch = 57.2958 * atan2(-R.at<double>(2,0), _sqrt);
	double yaw = 57.2958 * atan2(R.at<double>(1,0), R.at<double>(0,0));
	printf("roll, pitch, yaw:\n[%lf, %lf, %lf]\n", roll, pitch, yaw);

	while(1) {cv::waitKey(30);}
}

void VisualOdemetry::scale_calibration(cv::VideoCapture& camera)
{
	cv::Mat img1, img2;

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
		while(!camera.read(img1));
		imshow("scale calibration", img1);
		cv::waitKey(30);
	}

	printf("click the window to collect the calibration frame 2\n");
	calib_exit_signal = false;
	while(!calib_exit_signal) {
		while(!camera.read(img2));
		imshow("scale calibration", img2);
		cv::waitKey(30);
	}

	VOFeatures ref_frame_features, curr_frame_features;
	feature_detector.extract(img1, ref_frame_features);
	feature_detector.extract(img2, curr_frame_features);

	vector<DMatch> feature_matches;
	feature_detector.match(feature_matches, ref_frame_features, curr_frame_features);

	feature_detector.plot_matched_features(img1, img2, ref_frame_features,
                                               curr_frame_features, feature_matches);

	estimate_essential_matrix(ref_frame_features, curr_frame_features,
                                  feature_matches);
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
