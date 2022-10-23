#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;

int main() {

	// load image
	std::string img_name = "data\\data_m2\\1\\input\\in000095.jpg";
	cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR); // or IMREAD_GRAYSCALE

	if (!img.data) {
		return 1;
	}

	if (img.empty())
	{
		std::cout << "Could not read the image: " << img_name << std::endl;
		return 1;
	}

	imshow("Example Image", img);

	int wait = cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}