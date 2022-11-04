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

	for (int pos = 300; pos <= 1200; pos++) {
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
		
		
		//cv::subtract(in_img, gt_img, gt_img);
		//cv::subtract(in_img, gt_img, in_img);

		imshow("BG Substraction",  bgSub(in_img, gt_img));

		int wait = cv::waitKey(0);
		if (wait == 27) break; // ESC Key

		cv::destroyAllWindows();
	}
	return 0;
}