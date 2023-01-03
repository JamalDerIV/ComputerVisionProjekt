#include <opencv2\opencv.hpp>
#include <iostream>
#include <Windows.h>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;

class Detections {
public:
	float left, top, width, height, visibility, x, y, z;

	void setData(float l, float t, float w, float h, float v, float xC, float yC, float zC) {
		left = l;
		top = t;
		width = w;
		height = h;
		visibility = v;
		x = xC;
		y = yC;
		z = zC;
	}
};

class GroundTruth {
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
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050, "data\\data_m4\\1\\seqinfo.ini");
	GroundTruth (*gt)[200] = new GroundTruth[seqLength][200];
	Detections *det = new Detections[50];

	std::ifstream detfile("data\\data_m4\\1\\det\\det.txt");
	int frame, nDetections = 0;
	if (detfile >> frame);

	// Reading (custom) gt file
	//  -> the ',' seperator has been replaced with a ' ' whitespace
	std::string line;
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

	for (int pos = 1; pos <= seqLength; pos += 1) {
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

		std::cout << "\nFrame: " << pos << std::endl;
		// Reading detected values
		nDetections = 0;
		do {
			float id, left, top, width, height, visibility, x, y, z;
			detfile >> id >> left >> top >> width >> height >> visibility >> x >> y >> z;
			det[nDetections].setData(left, top, width, height, visibility, x, y, z);
			nDetections++;

			std::cout << left << " - " << top << std::endl;

			if (detfile) {
				detfile >> frame;
			}
			else {
				frame = 0;
			}
			
		} while (frame == pos);
		std::cout << "nDet " << nDetections << std::endl;

		//imshow("Image", in_img);

		std::cout << pos << std::endl;

		// do 10 steps before waiting again 
		if (pos % 1 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}

		cv::destroyAllWindows();
	}

	return 0;
}