#include <opencv2\opencv.hpp>
#include <iostream>
#include <Windows.h>

using namespace cv;

class Simple {
public:
	Simple() {
		std::cout << "Constructed" << std::endl;
	}
	
};

int main() {
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050, "data\\data_m4\\1\\seqinfo.ini");
	
	Simple* pSimple = new Simple[5][20];

	/*
	for (int pos = 1; pos <= 1050; pos += 1) {
		std::ostringstream in_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  "data\\data_m4\\1\\img1\\"<< pos_str <<".jpg";
		
		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);

		if (in_img.empty())
		{
			std::cout << "Could not read the image: " << in_img_name.str() << std::endl;
			return 1;
		}


		imshow("Image", in_img);

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
	*/
	return 0;
}