#include <opencv2\opencv.hpp>
#include <iostream>
#include <Windows.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;

/*
	The groundtruth files have been sorted with python code to ascend by frame
	and the differentiation by comma in the file has been replaced by a space
*/

Mat flaggedDetections;
Mat in_img;
Mat lastFrame;

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

	// Reduce this Objects Rect, to fit into targetRect
	Rect getReducedRect(Rect targetRect) {
		if (width > targetRect.width) {
			float l = left + ((width - targetRect.width) / 2);
			float t = top + ((height - targetRect.height) / 2);
			return Rect(l, t, targetRect.width, targetRect.height);
		}
		else {
			return Rect(left, top, width, height);
		}
		
	}
};

bool compareDetections(Detections d1, Detections d2) {
	return d1.getRect().area() < d2.getRect().area();
}

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
	Detections prevDet;
	int id, cutsRect, updated;
	float distance;
	std::vector<int> iouValues;

	TrackedObject(Detections d, int i, int c, int up) {
		det = d;
		prevDet = d;
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

	void updateDet(Detections newdet, int update) {
		prevDet = det;
		det = newdet;
		updated = update;
	}

	float getM() {
		return (det.top - prevDet.top) / (det.left - prevDet.left);
	}

	// returns <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>
	String getOutput() {
		std::ostringstream ret;
		ret << id << " " 
			<< det.left << " " 
			<< det.top << " " 
			<< det.width << " " 
			<< det.height << " " 
			<< det.confidence;
		return ret.str();
	}

};

// Returns 0 if template 1 has a better fit and 1 if template 2 is better
int compareTemplates(Mat roi, Mat templ1, Mat templ2) {
	Mat result1(roi.cols - templ1.cols + 1, roi.rows - templ1.rows + 1, CV_32FC1);
	Mat result2(roi.cols - templ2.cols + 1, roi.rows - templ2.rows + 1, CV_32FC1);
	matchTemplate(roi, templ1, result1, TM_CCOEFF_NORMED);  
	matchTemplate(roi, templ2, result2, TM_CCOEFF_NORMED);
	double minVal, val1, val2; // for SQDIFF / SQDIFF NORM use minimum, else maximum
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result1, &minVal, &val1, &minLoc, &maxLoc, Mat());
	minMaxLoc(result2, &minVal, &val2, &minLoc, &maxLoc, Mat());
	if (val1 > val2) return 0;
	return 1; //Point(matchLoc.x + templ1.cols, matchLoc.y + templ1.rows);
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

void calcMatrix(std::vector<TrackedObject> &trackedObjects, std::vector<Detections> detections){
	for (int i = 0; i < trackedObjects.size(); i++) {
		trackedObjects[i].iouValues.clear();
		for (int j = 0; j < detections.size();j++) {
			if (trackedObjects[i].updated == 1 || detections[j].ignore == 1) {
				//push 101 as a number that should be ignored
				trackedObjects[i].iouValues.push_back(101);
			}
			else {
				trackedObjects[i].iouValues.push_back(100 - (iou(trackedObjects[i].det.getRect(), detections[j].getRect()) * 100));
			}
		}
	}
}

//calculate the new position of the Tracked Object if we couldnt find a matching detection from the detection files
//TODO: den optischen Fluss berechnen und eine ungefähre Position berechnen
void calcNewPosition(TrackedObject &trackedObject) {
	std::cout << trackedObject.id << "same det" << std::endl;
	trackedObject.updateDet( trackedObject.det, 1); 
}

//adds a detection as a new Tracked Object to the vector array
void newTrackedObject(std::vector<TrackedObject> &trackedObjects, Detections &detection) {
	detection.ignore = 1;
	TrackedObject a(detection, trackedObjects.size() + 1, 0, 1);
	trackedObjects.push_back(a);
}

void assignNewDetection(std::vector<TrackedObject> &trackedObjects, std::vector<Detections> &detections) {
	calcMatrix(trackedObjects, detections);

	//assign a detection that cuts with no Tracked Object to a new Tracked Object (new person detected)
	for (int i = 0; i < detections.size(); i++) {
		if (detections[i].ignore == 1) {
			continue;
		}
		int amount = 0;
		//check the amount of overlays the detection has
		for (int j = 0; j < trackedObjects.size(); j++) {
			if (trackedObjects[j].updated == 1) {
				continue;
			}
			if (trackedObjects[j].iouValues[i] < 100) {
				amount++;
			}
		}

		//if the detection has no overlays, then get the detection out of the array and assign it to a new Tracked Object
		if (amount == 0) {
			newTrackedObject(trackedObjects, detections[i]);
		}
	}
}

/*call compareTemplates() to check which template of the first two fit better with the detection, if we have to check more than 2 templates,
* then check the next template with the winner of the last two until we have all checked and assign the detection to the best fitting template (Tracked Object)
*/
void checkAllTemplateMatching(std::vector<TrackedObject> &trackedObjects, std::vector<int> toCheck, Detections &detection) {
	int fittingPos = 999;

	for (int i = 1; i < toCheck.size(); i++) {
		if (!compareTemplates(in_img(detection.getRect()),
			lastFrame(trackedObjects[toCheck[i-1]].det.getReducedRect(detection.getRect())),
			lastFrame(trackedObjects[toCheck[i]].det.getReducedRect(detection.getRect())))) {

			std::cout << "Object " << trackedObjects[toCheck[i - 1]].id << " fits better." << std::endl;
			fittingPos = toCheck[i-1];
		}
		else {
			std::cout << "Object " << trackedObjects[toCheck[i]].id << " fits better." << std::endl;
			fittingPos = toCheck[i];
		}
	}

	if(fittingPos != 999){
		trackedObjects[fittingPos].updateDet(detection, 1);
		detection.ignore = 1;
	}
}

//recursively try to assign a detection for every still unassigned Tracked Object
void recursiveAssigning(std::vector<TrackedObject> &trackedObjects, std::vector<Detections> &detections){
	calcMatrix(trackedObjects, detections);
	std::vector<TrackedObject> tempObjects;
	std::vector<Detections> tempDetections;

	int count = 0;
	for (int j = 0; j < detections.size(); j++) {
		
		if (detections[j].ignore == 1) {
			continue;
		}

		std::vector<int> toCheck;

		for (int i = 0; i < trackedObjects.size(); i++) {
			if (trackedObjects[i].updated == 1) {
				continue;
			}

			if (trackedObjects[i].iouValues[j] < 100) {
				toCheck.push_back(i);
			}
		}


		if (toCheck.size() > 1) {
			count++;
			checkAllTemplateMatching(trackedObjects, toCheck, detections[j]);
			
			break;
		}
		toCheck.clear();
	}

	std::cout << "count : " << count << std::endl;

	int amountDet = 0;
	int amountObj = 0;
	for (int j = 0; j < detections.size(); j++) {
		if (detections[j].ignore == 1) {
			continue;
		}
		amountDet++;

		for (int i = 0; i < trackedObjects.size(); i++) {
			if (trackedObjects[i].updated == 1) {
				continue;
			}
			amountObj++;
		}
	}

	assignNewDetection(trackedObjects, detections);

	if (count > 0) {
		recursiveAssigning(trackedObjects, detections);
	}
	
}

void hungarian(int tries, std::vector<TrackedObject> &trackedObjects, std::vector<Detections> detections) {
	tries++;
	calcMatrix(trackedObjects, detections);
	
	for (int i = 0; i < trackedObjects.size(); i++) {
		int minNumber = 100;

		for (int j = 0; j < detections.size(); j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (trackedObjects[i].iouValues[j] < 50 && trackedObjects[i].iouValues[j] < minNumber && trackedObjects[i].iouValues[j] != 0) {
				minNumber = trackedObjects[i].iouValues[j];
			}
		}

		for (int j = 0; j < detections.size(); j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (trackedObjects[i].iouValues[j] < 100 && minNumber != 100) {
				trackedObjects[i].iouValues[j] -= minNumber;
			}
		}
	}

	for (int i = 0; i < trackedObjects.size(); i++) {
		int amount = 0, position = 0;

		//check amount of numbers under 30
		for (int j = 0; j < detections.size(); j++) {
			if (trackedObjects[i].iouValues[j] < 60) {
				amount++;
			}
		}

		//check amount of 0 and save the zeros position
		if (amount == 1) {
			amount = 0;
			for (int j = 0; j < detections.size(); j++) {
				if (trackedObjects[i].iouValues[j] == 0) {
					amount++;
					position = j;
				}
			}
		}

		//check if the detection previously found has an IoU of under 30 compared to all Tracked Objects
		if (amount == 1) {
			amount = 0;
			for (int j = 0; j < trackedObjects.size(); j++) {
				if (trackedObjects[j].iouValues[position] < 60) {
					amount++;
				}
			}
		}

		if (amount == 1) {
			//std::cout << "Row: " << i << " | " << ". row zero at: " << position << std::endl;
			trackedObjects[i].updateDet(detections[position], 1);
			detections[position].ignore = 1;
		}

	}

	//Tracked Object cuts with no new detection
	for (int i = 0; i < trackedObjects.size(); i++) {
		int amount = 0;
		if (trackedObjects[i].updated == 1) {
			continue;
		}

		for (int j = 0; j < detections.size(); j++) {
			if (detections[j].ignore == 1) {
				continue;
			}
			if (trackedObjects[i].iouValues[j] < 100) {
				amount++;
			}
		}

		if (amount == 0) {
			calcNewPosition(trackedObjects[i]); 
		}
	}


	assignNewDetection(trackedObjects, detections);

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
		//std::cout << "Try: " << tries << " | " << i+1 << ". row zeros: " << countZeros << std::endl;
		if (countZeros == 0) {
			everyRowZeros = false;
		}
	}


	if (!everyRowZeros && tries < 3) {
		hungarian(values, tries, trackedObjects, detections);
	}*/
}

int main() {
	const int dataset = 1;
	String filepath("data\\data_m4\\1\\");
	std::ostringstream seqinfoPath; seqinfoPath << "data\\data_m4\\" << dataset << "\\seqinfo.ini";
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050,  seqinfoPath.str().c_str());
	const int totalIDs = 150;

	GroundTruth *gt = new GroundTruth[totalIDs];
	std::vector<TrackedObject> trackedObjects;
	std::vector<Detections> det;

	// Reading (custom) gt and det files
	//  -> the ',' seperator has been replaced with a ' ' whitespace
	std::ifstream detfile(filepath+"det\\det.txt");
	std::ifstream gtfile(filepath + "gt\\gt_sorted.txt");
	int frame, nDetections = 0, nGroundtruths = 0;
	// lower/higher percentage border for Detections
	// this is applyed to the Median Detection
	const bool useMedianSizeFilter = false;
	float lower_p = 0.3, higher_p = 2;

	if (detfile >> frame);
	if (gtfile >> frame);

	// Writing tracked Object into file 
	std::ofstream outputFile;
	outputFile.open(filepath + "trackedOutput.txt");
	
	for (int pos = 1; pos <= seqLength; pos += 1) {
		std::ostringstream in_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  filepath+"img1\\"<< pos_str <<".jpg";
		in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);
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

		// Median Size Filter to filter out too big/small detections
		if (useMedianSizeFilter) {
			std::sort(det.begin(), det.end(), compareDetections);
			int median = det[(det.size() / 2) + 1].getRect().area();
			std::vector<Detections> temp_det;
			for (int d = 0; d < det.size(); d++) {
				if (det[d].getRect().area() >= median * lower_p
					&& det[d].getRect().area() <= median * higher_p)
				{
					temp_det.push_back(det[d]);
				}
			}
			det.clear();
			det = std::vector<Detections>(temp_det.size());
			std::copy(temp_det.begin(), temp_det.end(), det.begin());
			temp_det.clear();
		}
		

		//Reading Groundtruth values
		nGroundtruths = 0;
		do {
			float id, left, top, width, height, confidence, tag_class, visibility, z;
			if (dataset == 5) { 
				gtfile >> id >> left >> top >> width >> height >> confidence >> tag_class >> visibility >> z;
				gt[nGroundtruths].setData(left, top, width, height, confidence, visibility);
			}
			else {
				gtfile >> id >> left >> top >> width >> height >> confidence >> tag_class >> visibility;
				if (tag_class == 1) gt[nGroundtruths].setData(left, top, width, height, confidence, visibility);
			}
			
			nGroundtruths++;

			if (gtfile) {
				gtfile >> frame;
			}
			else {
				frame = 0;
			}

		} while (frame == pos);

		// push detections into trackedObjects
		if (pos <= 1) {
			for (int i = 0; i < det.size(); i++) {
				TrackedObject a(det[i], trackedObjects.size() + 1, 0, 0);
				trackedObjects.push_back(a);
				//cv::putText(in_img, std::to_string(a.id), { a.getX(), a.getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			}
		}

		for (int i = 0; i < det.size(); i++) {
			rectangle(imgCopy, det[i].getRect(), Scalar(255, 0, 0), 1);
			//cv::putText(in_img, "new", { trackedObjects[i].getX(), trackedObjects[i].getY() + 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			//std::cout << t.det.getRect() << " | id: " << t.id << std::endl;

		}
		lastFrame = in_img.clone();

		if (pos >= 2) {
			hungarian( 0, trackedObjects, det);
		}

		lastFrame = in_img.clone();
		/*
		if (pos >= 2 && dataset == 2) {
			Detections d_test;
			d_test.setData(880.f, 125.f, 55.f, 160.f, 0.4f,0);
			// if 2 tOs are close to each other and a det is lost in next frame, we compare the det to both tOs to find out which fits better 
			if (compareTemplates(in_img(d_test.getRect()),
				lastFrame(trackedObjects[9].det.getReducedRect(d_test.getRect())),
				lastFrame(trackedObjects[14].det.getReducedRect(d_test.getRect())))) {
				std::cout << "Object 15 fits." << std::endl;
			}
			else {
				std::cout << "Object 10 fits." << std::endl;
			}

		}
		*/

		//optical flow
		/*
		for (int i = 0; i < trackedObjects.size(); i++) {
			Mat lfGray, inGray;
			cvtColor(lastFrame, lfGray, COLOR_BGR2GRAY);
			cvtColor(in_img, inGray, COLOR_BGR2GRAY);
			calcOpticalFlowPyrLK(lfGray, inGray, trackedObjects[i].tlPoints[], p1, status, err, Size(sizeNumber, sizeNumber), maxLevel, criteria);
		}
		*/

		for (TrackedObject t : trackedObjects) {
			float m = t.getM();
			std::cout << "ID: " << t.id << " Direction: " << m << std::endl;
		}

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
			trackedObjects[i].updated = 0;
			trackedObjects[i].iouValues.clear();
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
		//imshow("Detections", imgCopy);
		// do 10 steps before waiting again 
		if (pos % 1 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}

		// write to output file
		for (int i = 0; i < trackedObjects.size(); i++)
			// <frame number>, tracked output... , <x>, <y>, <z>
			outputFile << pos << ' ' << trackedObjects[i].getOutput() << " -1 -1 -1" << std::endl;

		cv::destroyAllWindows();
	}

	return 0;
}