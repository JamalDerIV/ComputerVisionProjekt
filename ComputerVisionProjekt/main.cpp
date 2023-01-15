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

Mat flaggedDetections;

class Detections {
public:
	float left, top, width, height, confidence;
	int ignore;

	void setData(float l, float t, float w, float h, float c, int i) {
		left = l;
		top = t;
		width = w;
		height = h;
		confidence = c;
		ignore = i;
	}

	String print() {
		std::ostringstream ret;
		ret << left << " - " << top << " - " << width << " - " << height << " - " << confidence;
		return ret.str();
	}

	Rect getRect() {
		return Rect(left, top, width, height);
	}

	int getX() {
		return int(left);
	}
	int getY() {
		return int(top);
	}

	Rect roiRect() {
		float scale = 0.1;
		//float l = left * (1 - scale);
		//float t = top  * (1 - scale);
		//float w = width * (1 + scale);
		//float h = height * (1 + scale);
		float l = left - width;
		float t = top  - height/2;
		float w = width * 3;
		float h = height * 2;
		return Rect(l, t, w, h);
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
	int id, cutsRect, updated;
	

	TrackedObject(Detections d, int i, int c, int up) {
		det = d;
		id = i;
		cutsRect = c;
		updated = up;
	}
	int getX() {
		return int(det.left);
	}
	int getY() {
		return int(det.top);
	}

	void update(int up) {
		updated = up;
	}
};

// Returns 1 if template 1 has a better fit, 2 for template 2 and 0 if something went wrong
int findMatchingTemplate(Mat roi, Mat templ1, Mat templ2) {
	Mat result1(roi.cols - templ1.cols + 1, roi.rows - templ1.rows + 1, CV_32FC1);
	Mat result2(roi.cols - templ2.cols + 1, roi.rows - templ2.rows + 1, CV_32FC1);
	matchTemplate(roi, templ1, result1, TM_CCOEFF_NORMED);
	matchTemplate(roi, templ2, result2, TM_CCOEFF_NORMED);
	double minVal, val1, val2; // for SQDIFF / SQDIFF NORM use minimum, else maximum
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result1, &minVal, &val1, &minLoc, &maxLoc, Mat());
	std::cout << "Best Value 1: " << val1 << std::endl;
	minMaxLoc(result2, &minVal, &val2, &minLoc, &maxLoc, Mat());
	std::cout << "Best Value 2: " << val2 << std::endl;

	imshow("roi", roi);
	if (val1 > val2) return 1;
	return 2; //Point(matchLoc.x + templ1.cols, matchLoc.y + templ1.rows);
}

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

Mat calcMatrix(std::vector<TrackedObject> trackedObjects, std::vector<Detections> detections){
	std::vector<Detections> tempDet;
	std::vector<TrackedObject> tempTracked;
	for (Detections d : detections) {
		if (d.ignore == 0) {
			tempDet.push_back(d);
		}
	}

	for (TrackedObject t : trackedObjects) {
		if (t.updated == 0) {
			tempTracked.push_back(t);
		}
	}

	Mat values = cv::Mat::zeros(cv::Size( tempDet.size(), tempTracked.size()), CV_8UC1);
	flaggedDetections = cv::Mat::zeros(cv::Size( tempDet.size(), tempTracked.size()), CV_8UC1);
	int count=0;
	for (int i = 0; i < tempTracked.size(); i++) {
		for (int j = 0; j < tempDet.size();j++) {
			count++;
			//std::cout << "i: " << i << " " << trackedObjects[i].det.getRect() << " | j: " << j << " " << detections[j].getRect() << std::endl;
			//std::cout << count << " mal aufgerufen" << " tracked size: " << trackedObjects.size() << "det size: " << detections.size() << " | matrix size: " << values.size() << std::endl;
			values.at<uchar>(i, j) = 100 - (iou(tempTracked[i].det.getRect(), tempDet[j].getRect()) * 100);
		}
	}

	return values;
}

//calculate the new position of the Tracked Object if we couldnt find a matching detection from the detection files
//TODO: den optischen Fluss berechnen und eine ungefähre Position berechnen
void calcNewPosition(TrackedObject &trackedObject) {
	trackedObject.det = trackedObject.det;
	trackedObject.update(1);
}

//adds a detection as a new Tracked Object to the vector array
void newTrackedObject(std::vector<TrackedObject> &trackedObjects, Detections &detection) {
	detection.ignore = 1;
	TrackedObject a(detection, trackedObjects.size() + 1, 0, 1);
	trackedObjects.push_back(a);
}

//recursively try to assign a detection for every still unassigned Tracked Object
void recursiveAssigning(std::vector<TrackedObject> &trackedObjects, std::vector<Detections> &detections){
	Mat values = calcMatrix(trackedObjects, detections);
	bool changes = 0;

	//assign a detection that cuts with no Tracked Object to a new Tracked Object (new person detected)
	for (int i = 0; i < values.cols; i++) {
		int amount = 0;
		//check the amount of overlays the detection has
		for (int j = 0; j < values.rows; j++) {
			if (values.at<uchar>(j, i) < 100) {
				amount++;
			}
		}
		//if the detection has no overlays, then get the detection out of the array and assign it to a new Tracked Object
		if (amount == 0) {
			int zeroAmount = 0;
			for (int j = 0; j < detections.size(); j++) {
				if (detections[j].ignore == 0) {
					zeroAmount++;
				}

				if (zeroAmount == i + 1) {
					newTrackedObject(trackedObjects, detections[j]);
					changes = 1;
					break;
				}
			}
		}
	}

	if (changes) {
		values = calcMatrix(trackedObjects, detections);
	}



	std::cout << values << std::endl;
}

void hungarian(Mat values, int tries, std::vector<TrackedObject> &trackedObjects, std::vector<Detections> detections) {
	tries++;
	std::cout << values << std::endl;

	for (int i = 0; i < values.rows; i++) {
		int minNumber = 100;

		for (int j = 0; j < values.cols; j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (values.at<uchar>(i, j) < 40 && values.at<uchar>(i, j) < minNumber && values.at<uchar>(i, j) != 0) {
				minNumber = values.at<uchar>(i, j);
			}
		}

		for (int j = 0; j < values.cols; j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (values.at<uchar>(i, j) < 100 && minNumber != 100) {
				values.at<uchar>(i, j) -= minNumber;
			}
		}
	}

	std::cout << values << std::endl;

	for (int i = 0; i < values.rows; i++) {
		int amount = 0, position = 0;

		//check amount of numbers under 30
		for (int j = 0; j < values.cols; j++) {
			if (values.at<uchar>(i, j) < 30) {
				amount++;
			}
		}

		//check amount of 0 and save the zeros position
		if (amount == 1) {
			amount = 0;
			for (int j = 0; j < values.cols; j++) {
				if (values.at<uchar>(i, j) == 0) {
					amount++;
					position = j;
				}
			}
		}

		//check if the detection previously found has an IoU of under 30 compared to all Tracked Objects
		if (amount == 1) {
			amount = 0;
			for (int j = 0; j < values.rows; j++) {
				if (values.at<uchar>(j, position) < 30) {
					amount++;
				}
			}
		}

		if (amount == 1) {
			//std::cout << "Row: " << i << " | " << ". row zero at: " << position << std::endl;
			trackedObjects[i].det = detections[position];
			detections[position].ignore = 1;
			trackedObjects[i].update(1);
		}

	}


	//Tracked Object cuts with no new detection
	for (int i = 0; i < values.rows; i++) {
		int amount = 0;

		for (int j = 0; j < values.cols; j++) {
			if (detections[j].ignore == 1) {
				continue;
			}
			if (values.at<uchar>(i, j) < 100) {
				amount++;
			}
		}

		if (amount == 0) {
			calcNewPosition(trackedObjects[i]);
		}
	}

	for (TrackedObject t : trackedObjects) {
		if (t.updated == 0) {
			std::cout << t.id << std::endl;
		}
	}

	recursiveAssigning(trackedObjects, detections);

	/*for (int i = 0; i < flaggedDetections.rows; i++) {
		for (int j = 0; j < flaggedDetections.cols; j++) {
			if (flaggedDetections.at<uchar>(i, j) == 1) {
				trackedObjects[i].det = tempDetections[j];
				detections[j].update(1);
			}
		}
	}*/

	/*bool everyRowZeros = true;
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
		hungarian(values, tries, trackedObjects, detections);
	}*/
}

int main() {
	String filepath("data\\data_m4\\1\\");
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050, "data\\data_m4\\1\\seqinfo.ini");
	const int totalIDs = 150;

	GroundTruth *gt = new GroundTruth[totalIDs];
	std::vector<TrackedObject> trackedObjects;
	std::vector<Detections> det;
	Mat lastFrame;

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

		cv::Mat imgCopy = in_img.clone();

		det.clear();

		// Reading detected values
		do {
			float id, left, top, width, height, confidence, x, y, z;
			detfile >> id >> left >> top >> width >> height >> confidence >> x >> y >> z;
			Detections dett;
			if (confidence > 0) {
				dett.setData(left, top, width, height, confidence,0);
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

		if (pos <= 1) {
			// push detections into trackedObjects
			for (int i = 0; i < det.size(); i++) {
				TrackedObject a(det[i], trackedObjects.size() + 1, 0, 0);
				trackedObjects.push_back(a);
				//cv::putText(in_img, std::to_string(a.id), { a.getX(), a.getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			}
		}
		
		/*lastFrame = in_img.clone();
		if (pos > 2) {
			Detections d_test;
			d_test.setData(861.f, 118.f, 79.f, 178.f, 0.4f, 0);

			imshow("tO 10", lastFrame(trackedObjects[9].det.getRect()));
			imshow("tO 25", lastFrame(trackedObjects[14].det.getRect()));

			int res = findMatchingTemplate(in_img(d_test.roiRect()), lastFrame(trackedObjects[9].det.getRect()), lastFrame(trackedObjects[14].det.getRect()));
			std::cout << "winner : : : " << res << std::endl;
		}*/


		for (int i = 0; i < det.size(); i++) {
			rectangle(imgCopy, det[i].getRect(), Scalar(255, 0, 0), 1);
			//cv::putText(in_img, "new", { trackedObjects[i].getX(), trackedObjects[i].getY() + 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			//std::cout << t.det.getRect() << " | id: " << t.id << std::endl;

		}

		if (pos >= 2) {
			Mat values = calcMatrix(trackedObjects, det);
			hungarian(values, 0, trackedObjects, det);
		}


		//std::cout << trackedObjects[6].det.getRect() << std::endl;


		//draw tracked Objects
		for (int i = 0; i < trackedObjects.size(); i++) {
			if (trackedObjects[i].updated == 1) {
				rectangle(in_img, trackedObjects[i].det.getRect(), Scalar(0, 0, 255), 1);
				cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,0,255 }, 2);
			}
			else {
				rectangle(in_img, trackedObjects[i].det.getRect(), Scalar(0, 255, 0), 1);
				cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			}
			trackedObjects[i].update(0);
			//std::cout << t.det.getRect() << " | id: " << t.id << std::endl;
			
		}

		//trackedObjects.clear();

		//draw groundtruths
		/*
		for (int i = 1; i < nGroundtruths; i++) {
			rectangle(in_img, Rect(gt[i].left, gt[i].top, gt[i].width, gt[i].height), Scalar(255, 0, 0), 1);
			//std::cout << gt[i].left << " - " << gt[i].top << " - " << gt[i].width << " - " << gt[i].height << " - " << std::endl;
		}*/
		

		imshow("TrackedObjects", in_img);
		imshow("Detections", imgCopy);
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