#include <opencv2\opencv.hpp>
#include <iostream>
#include <Windows.h>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;

/*
	The groundtruth files have been sorted with python code to ascend by frame
	and the differentiation by comma has been replaced by a space
*/

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

	String print() {
		std::ostringstream ret;
		ret << left << " - " << top << " - " << width << " - " << height << " - " << visibility;
		return ret.str();
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

	String print() {
		std::ostringstream ret;
		ret << left << " - " << top << " - " << width << " - " << height;
		return ret.str();
	}
};


int main() {
	String filepath("data\\data_m4\\3\\");
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050, "data\\data_m4\\3\\seqinfo.ini");

	GroundTruth *gt = new GroundTruth[150];
	Detections *det = new Detections[150];

	// Reading (custom) gt and det files
	//  -> the ',' seperator has been replaced with a ' ' whitespace
	std::ifstream detfile(filepath+"det\\det.txt");
	std::ifstream gtfile(filepath + "gt\\gt_sorted.txt");
	int frame, nDetections = 0, nGroundtruths = 0;
	
	if (detfile >> frame);
	if (gtfile >> frame);
	
	for (int pos = 1; pos <= seqLength; pos += 1) {
		std::ostringstream in_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  filepath+"img1\\"<< pos_str <<".jpg";
		
		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);

		if (in_img.empty())
		{
			std::cout << "Could not read the image: " << in_img_name.str() << std::endl;
			return 1;
		}

		// Reading detected values
		nDetections = 0;
		do {
			float id, left, top, width, height, visibility, x, y, z;
			detfile >> id >> left >> top >> width >> height >> visibility >> x >> y >> z;
			det[nDetections].setData(left, top, width, height, visibility, x, y, z);
			nDetections++;

			if (detfile) {
				detfile >> frame;
			}
			else {
				frame = 0;
			}
			
		} while (frame == pos);

		nGroundtruths = 0;
		do {
			float id, left, top, width, height, x, y, z;
			gtfile >> id >> left >> top >> width >> height >> x >> y >> z;
			gt[nGroundtruths].setData(left, top, width, height, x, y, z);
			nGroundtruths++;

			if (gtfile) {
				gtfile >> frame;
			}
			else {
				frame = 0;
			}

		} while (frame == pos);

		//draw detections
		for (int i = 0; i < nDetections; i++) {
			rectangle(in_img, Rect(det[i].left, det[i].top, det[i].width, det[i].height), Scalar(0, 255, 0), 1);
		}

		//draw groundtruths
		for (int i = 1; i < nGroundtruths; i++) {
			if (gt[i].left > 0) {
				rectangle(in_img, Rect(gt[i].left, gt[i].top, gt[i].width, gt[i].height), Scalar(255, 0, 0), 1);
			}
			//std::cout << gt[i].left << " - " << gt[i].top << " - " << gt[i].width << " - " << gt[i].height << " - " << std::endl;
		}

		imshow("Image", in_img);

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