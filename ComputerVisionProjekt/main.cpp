#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;

cv::Mat bgSub(cv::Mat img1, cv::Mat img2) {
	cv::Mat solution;
	solution.create(img1.rows, img1.cols, CV_8UC3);
	for (int row = 0; row < solution.rows; row++) {
		for (int col = 0; col < solution.cols; col++) {
			cv::Vec3b whitePixel = { 255, 255, 255 };
			solution.at<cv::Vec3b>(row, col) = img1.at<cv::Vec3b>(row, col) - (whitePixel - img2.at<cv::Vec3b>(row, col));
		}
	}
	return solution;
}

int main() {

	cv::Ptr<BackgroundSubtractor> mog2BS = createBackgroundSubtractorMOG2();
	cv::Ptr<BackgroundSubtractor> knnBS = createBackgroundSubtractorKNN();

	for (int pos = 300; pos <= 1200; pos += 10) { // Stepping 10 Frames at a time might conflict with Background Subtraction
		std::ostringstream in_img_name, gt_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  "data\\data_m2\\1\\input\\in"<< pos_str <<".jpg";
		gt_img_name <<  "data\\data_m2\\1\\groundtruth\\gt"<< pos_str <<".png";
		
		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);
		cv::Mat gt_img = cv::imread(gt_img_name.str(), cv::IMREAD_COLOR);

		if (in_img.empty())
		{
			std::cout << "Could not read the image: " << in_img_name.str() << std::endl;
			return 1;
		}
		if (gt_img.empty())
		{
			std::cout << "Could not read the image: " << gt_img_name.str() << std::endl;
			return 1;
		}

		cv::Mat mog2Mask, knnMask;
		mog2BS->apply(in_img, mog2Mask);
		knnBS->apply(in_img, knnMask);

		imshow("Own Substraction",  bgSub(in_img, gt_img));
		//imshow("Double Substraction",  in_img - (in_img - gt_img));
		imshow("Mog2 Background Substraction",  mog2Mask);
		imshow("KNN Background Substraction",  knnMask);

		int wait = cv::waitKey(0);
		if (wait == 27) break; // ESC Key

		cv::destroyAllWindows();
	}
	return 0;
}