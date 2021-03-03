#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/* debug images */
Mat keypoint_img_last, keypoint_img_now;
Mat feature_match_img, filtered_feature_match_img;

void feature_matching(Mat& frame1, Mat& frame2)
{
}

int main()
{
	cv::Mat camera;
	cv::VideoCapture cap(0, cv::CAP_V4L);

	if(cap.isOpened() != true) {
		cout << "failed to open the camera.\n";
		return -1;
	}

	std::vector<KeyPoint> keypoints_last, keypoints_now;
	Mat descriptors1, descriptors2;
	Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

	Mat img_last, img_now;

	/* initialization */
	while(!cap.read(camera));
	img_last = cv::Mat(camera);

	while(cap.read(camera)) {
		img_now = cv::Mat(camera);

		/* extract orb features */
		orb->detect(img_last, keypoints_last);
		orb->detect(img_now, keypoints_now);
		orb->compute(img_last, keypoints_last, descriptors1);
		orb->compute(img_now, keypoints_now, descriptors2);

		/* match orb features */
		vector<DMatch> feature_matches;
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(descriptors1, descriptors2, feature_matches);

		/* record min/max feature distance */
		double min_dist = 10000, max_dist = 0;
		for(int i = 0; i < descriptors1.rows; i++) {
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
		for(int i = 0; i < descriptors1.rows; i++) {
			if(feature_matches[i].distance <= max(2 * min_dist, 30.0)) {
				good_feature_matches.push_back(feature_matches[i]);
			}
		}

		/* debug visualizations */
		drawKeypoints(img_last, keypoints_last, keypoint_img_last,
		              Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		drawKeypoints(img_now, keypoints_now, keypoint_img_now,
		              Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("keypoints of last image", keypoint_img_last);
		imshow("keypoints of current image", keypoint_img_now);

		drawMatches(img_last, keypoints_last, img_now, keypoints_now,
		            feature_matches, feature_match_img);
		drawMatches(img_last, keypoints_last, img_now, keypoints_now,
		            good_feature_matches, filtered_feature_match_img);
		imshow("original matched features", feature_match_img);
		imshow("filtered good matched features", filtered_feature_match_img);
		cv::waitKey(30);

		img_last = img_now;
	}

	return 0;
}
