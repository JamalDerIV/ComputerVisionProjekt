#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;

class Detections {
public:
	float left, top, width, height, x, y, z;

	void setData(float l, float t, float w, float h, float xC, float yC, float zC) {
		left = l;
		top = t;
		width = w;
		height = h;
		x = xC;
		y = yC;
		z = zC;
	}
};


int main() {
	Detections (*gt)[200] = new Detections[600][200];

	// Reading (custom) gt file
	//  -> the ',' seperator has been replaced with a ' ' whitespace
	/*std::string line;
	std::ifstream gtfile("data\\data_m4\\1\\gt\\gt.txt");

	while (std::getline(gtfile, line)) {
		std::istringstream iss(line);
		int frame, id, bb_left, bb_top, bb_width, bb_height, x, y, z;
		if (!(iss >> frame >> id >> bb_left >> bb_top >> bb_width >> bb_height >> x >> y >> z)) {
			break;
		}
		gt[frame][id].setData(bb_left, bb_top, bb_width, bb_height, x, y, z);
	}
	
	if (!gtfile.is_open()) {
		std::cout << "Could not open Ground Truth file. Evaluation will be skipped." << std::endl;
	}
	*/

	/*for (int pos = 1; pos <= 1050; pos += 1) {
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
		if (pos % 1 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}

		cv::destroyAllWindows();
	}*/

	return 0;
}