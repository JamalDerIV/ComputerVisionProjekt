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
	float left, top, width, height, confidence, visibility;

	void setData(float l, float t, float w, float h, float c, float v) {
		left = l;
		top = t;
		width = w;
		height = h;
		confidence = c;
		visibility = v;
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
	int id, cutsRect;
	TrackedObject(Detections d, int i,int c) {
		det = d;
		id = i;
		cutsRect = c;
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

void hungarian(Mat values, int tries) {
	tries++;
	std::cout << values << std::endl;

	for (int i = 0; i < values.rows; i++) {
		int minNumber = 100;

		for (int j = 0; j < values.cols; j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (values.at<uchar>(i, j) < 100 && values.at<uchar>(i, j) < minNumber && values.at<uchar>(i, j) != 0) {
				minNumber = values.at<uchar>(i, j);
			}
		}

		for (int j = 0; j < values.cols; j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (values.at<uchar>(i, j) < 100 && minNumber != 100) {
				if (values.at<uchar>(i, j) == 0) {
					continue;
				}
				values.at<uchar>(i, j) -= minNumber;
			}
		}
	}

	std::cout << values << std::endl;

	bool everyRowZeros = true;
	for (int i = 0; i < values.rows; i++) {
		int countZeros=0;
		for (int j = 0; j < values.cols; j++) {
			if (values.at<uchar>(i, j) == 0) {
				countZeros++;
			}
		}
		std::cout << "Try: " << tries << " | " << i+1 << ". row zeros: " << countZeros << std::endl;
		if (countZeros == 0) {
			everyRowZeros = false;
		}
	}
	

	if (!everyRowZeros && tries < 3) {
		hungarian(values, tries);
	}
}

void calcMatrix(std::vector<TrackedObject> trackedObjects, std::vector<Detections> detections){
	Mat values = cv::Mat::zeros(cv::Size( detections.size(), trackedObjects.size()), CV_8UC1);
	int count=0;
	for (int i = 0; i < trackedObjects.size(); i++) {
		for (int j = 0; j < detections.size();j++) {
			count++;
			//std::cout << "i: " << i << " " << trackedObjects[i].det.getRect() << " | j: " << j << " " << detections[j].getRect() << std::endl;
			//std::cout << count << " mal aufgerufen" << " tracked size: " << trackedObjects.size() << "det size: " << detections.size() << " | matrix size: " << values.size() << std::endl;
			values.at<uchar>(i,j) = 100-(iou(trackedObjects[i].det.getRect(), detections[j].getRect())*100);
		}
	}
	
	hungarian(values,0);
}

int main() {
	String filepath("data\\data_m4\\2\\");
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050, "data\\data_m4\\2\\seqinfo.ini");
	const int totalIDs = 150;

	GroundTruth *gt = new GroundTruth[totalIDs];
	std::vector<TrackedObject> trackedObjects;
	std::vector<Detections> det;

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

		det.clear();

		// Reading detected values
		do {
			float id, left, top, width, height, confidence, x, y, z;
			detfile >> id >> left >> top >> width >> height >> confidence >> x >> y >> z;
			Detections dett;
			if (confidence > 0) {
				dett.setData(left, top, width, height, confidence);
				det.push_back(dett);
			}

			if (detfile) {
				detfile >> frame;
			}
			else {
				frame = 0;
			}
			
		} while (frame == pos);

		nGroundtruths = 0;
		do {
			float id, left, top, width, height, confidence, tag_class, visibility;
			gtfile >> id >> left >> top >> width >> height >> confidence >> tag_class >> visibility;
			if (tag_class == 1) gt[nGroundtruths].setData(left, top, width, height, confidence, visibility);
			nGroundtruths++;

			if (gtfile) {
				gtfile >> frame;
			}
			else {
				frame = 0;
			}

		} while (frame == pos);

		// push detections into trackedObjects
		if (trackedObjects.size() <= 0) {
			for (int i = 0; i < det.size(); i++) {
				TrackedObject a(det[i], trackedObjects.size() + 1, 0);
				trackedObjects.push_back(a);
				//cv::putText(in_img, std::to_string(a.id), { a.getX(), a.getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			}
		}

		//draw tracked Objects
		for (int i = 0; i < trackedObjects.size(); i++) {
			rectangle(in_img, trackedObjects[i].det.getRect(), Scalar(0, 255, 0), 1);
			cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			//std::cout << trackedObjects[i].det.getRect() << " | id: " << trackedObjects[i].id << std::endl;
			
		}
		
		if (pos >= 2) {
			calcMatrix(trackedObjects, det); 
		}

		//trackedObjects.clear();

		//draw groundtruths
		/*
		for (int i = 1; i < nGroundtruths; i++) {
			rectangle(in_img, Rect(gt[i].left, gt[i].top, gt[i].width, gt[i].height), Scalar(255, 0, 0), 1);
			//std::cout << gt[i].left << " - " << gt[i].top << " - " << gt[i].width << " - " << gt[i].height << " - " << std::endl;
		}*/
		

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