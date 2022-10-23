#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;

int main() {

	// load image

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
		
		
		cv::subtract(in_img, gt_img, gt_img);
		cv::subtract(in_img, gt_img, in_img);
		imshow(pos_str, in_img);

		int wait = cv::waitKey(0);
		cv::destroyAllWindows();
	}
	

	return 0;
}