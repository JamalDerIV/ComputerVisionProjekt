#include <opencv2\opencv.hpp>
#include <iostream>
#include <Windows.h>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;

/*
	The groundtruth files have been sorted with python code to ascend by frame
	and the differentiation by comma in the file has been replaced by a space
*/

class Detections {
public:
	float left, top, width, height, confidence;

	void setData(float l, float t, float w, float h, float c) {
		left = l;
		top = t;
		width = w;
		height = h;
		confidence = c;
	}

	String print() {
		std::ostringstream ret;
		ret << left << " - " << top << " - " << width << " - " << height << " - " << confidence;
		return ret.str();
	}

	Rect getRect() {
		return Rect(left, top, width, height);
	}
};

class GroundTruth {
public:
	float left, top, width, height;

	void setData(float l, float t, float w, float h) {
		left = l;
		top = t;
		width = w;
		height = h;
	}

	String print() {
		std::ostringstream ret;
		ret << left << " - " << top << " - " << width << " - " << height;
		return ret.str();
	}
};

class TrackedObject {
public:
	Detections det;
	int id;
	TrackedObject(Detections d, int i) {
		det = d;
		id = i;
	}
	int getX() {
		return int(det.left);
	}
	int getY() {
		return int(det.top);
	}
};

double iou(Rect boxGT, Rect boxPredicted) {
	int xA = max(boxGT.x, boxPredicted.x),
		xB = min(boxGT.x + boxGT.width, boxPredicted.x + boxPredicted.width),
		yA = max(boxGT.y, boxPredicted.y),
		yB = min(boxGT.y + boxGT.height, boxPredicted.y + boxPredicted.height);
	if (xA > xB) return 0.0;
	if (yA > yB) return 0.0;
	// intersection
	int iArea = (xB - xA + 1) * (yB - yA + 1);
	//union
	int areaA = (boxGT.width + 1) * (boxGT.height + 1);
	int areaB = (boxPredicted.width + 1) * (boxPredicted.height + 1);

	return (double)iArea / (double)(areaA + areaB - iArea);
};

int main() {
	String filepath("data\\data_m4\\2\\");
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050, "data\\data_m4\\2\\seqinfo.ini");
	const int totalIDs = 150;

	GroundTruth *gt = new GroundTruth[totalIDs];
	Detections *det = new Detections[totalIDs];
	std::vector<TrackedObject> trackedObjects;

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
			float id, left, top, width, height, confidence, x, y, z;
			detfile >> id >> left >> top >> width >> height >> confidence >> x >> y >> z;
			det[nDetections].setData(left, top, width, height, confidence);
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
			gt[nGroundtruths].setData(left, top, width, height);
			nGroundtruths++;

			if (gtfile) {
				gtfile >> frame;
			}
			else {
				frame = 0;
			}

		} while (frame == pos);

		

		// Hungarian Method
		if (trackedObjects.size() == 0) {
			for (int i = 0; i < nDetections; i++) {
				TrackedObject a(det[i], trackedObjects.size() + 1);
				trackedObjects.push_back(a);
				//cv::putText(in_img, std::to_string(a.id), { a.getX(), a.getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			}
		}
		else {
			// nDetextions x trackedObjects
			std::vector<std::vector<double>> iouMatrix(trackedObjects.size(), std::vector<double>(nDetections, 0));
			for (int t = 0; t < trackedObjects.size(); t++) {
				for (int d = 0; d < nDetections; d++) {
					iouMatrix.at(t).at(d) = iou(trackedObjects[t].det.getRect(), det[d].getRect());
				}
			}

			for (int t = 0; t < iouMatrix.size(); t++) {
				double highestValue = 0;
				int pos = 0;
				for (int d = 0; d < nDetections; d++) {
					if (highestValue < iouMatrix.at(t).at(d)) {
						highestValue = iouMatrix.at(t).at(d);
						pos = d;
					}
				}
				trackedObjects.at(t).det = det[pos];
				for (int d = t; d < iouMatrix.size(); d++) {
					iouMatrix.at(t).at(pos) = 0;
				}

			}
			
		}

		//draw detected
		for (int i = 0; i < nDetections; i++) {
			rectangle(in_img, det[i].getRect(), Scalar(0, 0, 255), 1);
			//cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
		}

		//draw tracked Objects
		for (int i = 0; i < trackedObjects.size(); i++) {
			rectangle(in_img, trackedObjects[i].det.getRect(), Scalar(0, 255, 0), 1);
			cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
		}


		//draw groundtruths
		/*
		for (int i = 1; i < nGroundtruths; i++) {
			rectangle(in_img, Rect(gt[i].left, gt[i].top, gt[i].width, gt[i].height), Scalar(255, 0, 0), 1);
			//std::cout << gt[i].left << " - " << gt[i].top << " - " << gt[i].width << " - " << gt[i].height << " - " << std::endl;
		}
		*/

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