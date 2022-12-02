#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;


int main() {

	cv::Ptr<BackgroundSubtractorMOG2> mog2BS = createBackgroundSubtractorMOG2();
	
	mog2BS->setShadowThreshold(0.82);


	for (int pos = 19; pos <= 1050; pos += 1) {
		std::ostringstream in_img_name, gt_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  "data\\data_m3\\1\\img1\\"<< pos_str <<".jpg";
		
		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);

		if (in_img.empty())
		{
			std::cout << "Could not read the image: " << in_img_name.str() << std::endl;
			return 1;
		}

		cv::Mat mog2Mask;
		mog2BS->apply(in_img, mog2Mask);

		imshow("Mog2 Background Substraction",  mog2Mask);

		std::cout << pos << std::endl;

		// do 10 steps before waiting again 
		if (pos % 10 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}

		cv::destroyAllWindows();
	}
	return 0;
}