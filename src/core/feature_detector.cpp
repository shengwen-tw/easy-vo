#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "feature_detector.hpp"

using namespace std;
using namespace cv;

VOFeatureDetector::VOFeatureDetector()
{
	orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
}

void VOFeatureDetector::extract(Mat& img, VOFeatures& features)
{
	/* extract image feature keypoints and generate descriptors */
	orb->detect(img, features.keypoints);
	orb->compute(img, features.keypoints, features.descriptors);
}

void VOFeatureDetector::match(vector<DMatch>& feature_matches,
                              VOFeatures& features1, VOFeatures& features2)
{
	/* match orb features */
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(features1.descriptors, features2.descriptors, feature_matches);

	/* record min/max feature distance */
	double min_dist = 10000, max_dist = 0;
	for(int i = 0; i < features1.descriptors.rows; i++) {
		double curr_dist = feature_matches[i].distance;
		if(curr_dist < min_dist) {
			min_dist = curr_dist;
		}

		if(curr_dist > max_dist) {
			max_dist = min_dist;
		}
	}
	//printf("feature dist: min:%lf, max:%lf\n", min_dist, max_dist);

	/* matched feature filtering */
	std::vector<DMatch> good_feature_matches;
	for(int i = 0; i < features1.descriptors.rows; i++) {
		if(feature_matches[i].distance <= max(2 * min_dist, 30.0)) {
			good_feature_matches.push_back(feature_matches[i]);
		}
	}
}

void VOFeatureDetector::plot_matched_features(Mat& img1, Mat& img2,
                VOFeatures& features1, VOFeatures& features2,
                vector<DMatch>& feature_matches)
{
	Mat keypoints_img1, keypoints_img2;
	Mat match_img;

	drawKeypoints(img1, features1.keypoints, keypoints_img1,
	              Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, features2.keypoints, keypoints_img2,
	              Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("keypoints of image1", keypoints_img1);
	imshow("keypoints of image2", keypoints_img2);

	drawMatches(img1, features1.keypoints, img2, features2.keypoints,
	            feature_matches, match_img);
	imshow("matched features", match_img);
	cv::waitKey(30);
}
