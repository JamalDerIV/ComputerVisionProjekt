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
int imWidth=0, imHeight=0;

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

	Rect getRectForCutting() {
		int tempWidth = 0, tempHeight = 0,tempLeft = 0,tempTop = 0;
		if (left > imWidth) {
			left = imWidth;
		}
		if (top > imHeight) {
			top = imHeight;
		}
		if (left + width >= imWidth) {
			width = imWidth - left;
		}
		if (top + height >= imHeight) {
			height = imHeight - top;
		}
		if (left < 0) {
			left = 0;
		}
		if (top < 0) {
			top = 0;
		}
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
		if (width > targetRect.width || height > targetRect.height) {
			float l = left + ((width - targetRect.width) / 2);
			float t = top + ((height - targetRect.height) / 2);

			if (l >= imWidth) {
				l = imWidth - 1;
			}
			if (l + targetRect.width > imWidth) {
				targetRect.width = imWidth - l;
			}
			if (t >= imHeight) {
				t = imHeight - 1;
			}
			if (t + targetRect.height > imHeight) {
				targetRect.height = imHeight - t;
			}
			if (l < 0) {
				l = 0;
			}
			if (t < 0) {
				t = 0;
			}
			if (targetRect.width < 1) {
				targetRect.width = 1;
			}
			if (targetRect.height < 1) {
				targetRect.height = 1;
			}
			return Rect(l, t, targetRect.width, targetRect.height);
		}
		else {
			if (left >= imWidth) {
				left = imWidth - 1;
			}
			if (left + width >= imWidth) {
				width = imWidth - left;
			}
			if (top >= imHeight) {
				top = imHeight - 1;
			}
			if (top + height >= imHeight) {
				height = imHeight - top;
			}
			if (left < 0) {
				left = 0;
			}
			if (top < 0) {
				top = 0;
			}
			if (width < 1) {
				width = 1;
			}
			if (height < 1) {
				height = 1;
			}

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
	int id, cutsRect, updated, noNewDetection, ignore;
	std::vector<float> movementLeft, movementTop;
	std::vector<int> iouValues;

	TrackedObject(Detections d, int i, int c, int up) {
		det = d;
		prevDet = d;
		id = i;
		cutsRect = c;
		updated = up;
		ignore = 0;
		noNewDetection = 0;
	}

	int getX() {
		return int(det.left);
	}

	int getY() {
		return int(det.top);
	}

	void updateDet(Detections newdet, int update) {
		noNewDetection = 0;
		prevDet = det;
		det = newdet;
		updated = update;
		updateMovement();
	}

	void updateMovement() {
		float mleft = (det.left + (det.width * 0.5)) - (prevDet.left + (prevDet.width * 0.5));
		float mtop = (det.top + (det.height * 0.5)) - (prevDet.top + (prevDet.height * 0.5));
		if (movementLeft.size() >= 5) movementLeft.erase(movementLeft.begin());
		movementLeft.push_back(mleft);
		if (movementTop.size() >= 5) movementTop.erase(movementTop.begin());
		movementTop.push_back(mtop);
	}

	void move() {
		noNewDetection++;
		
		float averageTop = 0;
		float averageLeft = 0;

		if (movementTop.size() > 0) {
			for (float val : movementTop) averageTop += val;
			averageTop /= movementTop.size();
		}

		if (movementLeft.size() > 0) {
			for (float val : movementLeft) averageLeft += val;
			averageLeft /= movementLeft.size();
		}

		if (det.top + averageTop <= imHeight) {
			det.top += averageTop;
		}
		if (det.left + averageLeft <= imWidth) {
			det.left += averageLeft;
		}
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
			if (trackedObjects[i].updated == 1 || trackedObjects[i].ignore == 1 || detections[j].ignore == 1) {
				//push 101 as a number that should be ignored
				trackedObjects[i].iouValues.push_back(101);
			}
			else {
				trackedObjects[i].iouValues.push_back(100 - (iou(trackedObjects[i].det.getRect(), detections[j].getRect()) * 100));
			}
		}
	}
}

//temp delete later
Mat calcMatrixo(std::vector<TrackedObject>& trackedObjects, std::vector<Detections> detections) {
	Mat values = cv::Mat::zeros(cv::Size(detections.size(), trackedObjects.size()), CV_8UC1);
	for (int i = 0; i < trackedObjects.size(); i++) {
		trackedObjects[i].iouValues.clear();
		for (int j = 0; j < detections.size();j++) {
			
			values.at<uchar>(i,j) = (100 - (iou(trackedObjects[i].det.getRect(), detections[j].getRect()) * 100));
			
		}
	}
	return values;
}


//calculate the new position of the Tracked Object if we couldnt find a matching detection from the detection files
void calcNewPosition(TrackedObject &trackedObject) {
	trackedObject.move();
	trackedObject.updated = 1;
}

//adds a detection as a new Tracked Object to the vector array
void newTrackedObject(std::vector<TrackedObject> &trackedObjects, Detections &detection) {
	detection.ignore = 1;
	TrackedObject a(detection, trackedObjects.size() + 1, 0, 1);
	trackedObjects.push_back(a);
}

void updateTrackedObject(TrackedObject &trackedObject, Detections &detection) {
	if (abs(trackedObject.det.height - detection.height) > trackedObject.det.height * 0.2 ||
		abs(trackedObject.det.width - detection.width) > trackedObject.det.width * 0.2) {
		calcNewPosition(trackedObject);
	}
	else {
		trackedObject.updateDet(detection, 1);
		detection.ignore = 1;
	}

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
			if (trackedObjects[j].updated == 1 || trackedObjects[j].ignore == 1) {
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
	int lastFit = 0;

	

	for (int i = 1; i < toCheck.size(); i++) {
		if (!compareTemplates(in_img(detection.getRectForCutting()),
			lastFrame(trackedObjects[toCheck[lastFit]].det.getReducedRect(detection.getRectForCutting())),
			lastFrame(trackedObjects[toCheck[i]].det.getReducedRect(detection.getRectForCutting())))) {

			fittingPos = toCheck[lastFit];
		}
		else {
			fittingPos = toCheck[i];
			lastFit = i;
		}
	}

	if(fittingPos != 999){
		updateTrackedObject(trackedObjects[fittingPos], detection);
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
			if (trackedObjects[i].updated == 1 || trackedObjects[i].ignore == 1) {
				continue;
			}

			if (trackedObjects[i].iouValues[j] < 50) {
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

	int amountDet = 0;
	int amountObj = 0;
	for (int j = 0; j < detections.size(); j++) {
		if (detections[j].ignore == 1) {
			continue;
		}
		amountDet++;

		for (int i = 0; i < trackedObjects.size(); i++) {
			if (trackedObjects[i].updated == 1 || trackedObjects[i].ignore == 1) {
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

void assignLastDetections(std::vector<TrackedObject> &trackedObjects, std::vector<Detections> &detections) {
	calcMatrix(trackedObjects,detections);
	for (int i = 0; i < trackedObjects.size();i++) {
		if (trackedObjects[i].updated == 1 || trackedObjects[i].ignore == 1) {
			continue;
		}

		int lowestValue = 101;
		int pos = 999;
		for (int j = 0; j < detections.size(); j++) {
			if (detections[j].ignore == 1) {
				continue;
			}

			if (lowestValue > trackedObjects[i].iouValues[j] && trackedObjects[i].iouValues[j] < 60 && abs(lowestValue- trackedObjects[i].iouValues[j]) >= 5) {
				lowestValue = trackedObjects[i].iouValues[j];
				pos = j;
			}
			else if(lowestValue > trackedObjects[i].iouValues[j] && trackedObjects[i].iouValues[j] < 60 && abs(lowestValue - trackedObjects[i].iouValues[j]) < 4){
				if (pos == 999) {
					continue;
				}

				if (compareTemplates(in_img(detections[j].getRectForCutting()),
					lastFrame(trackedObjects[pos].det.getReducedRect(detections[j].getRectForCutting())),
					lastFrame(trackedObjects[i].det.getReducedRect(detections[j].getRectForCutting())))) {

					lowestValue = trackedObjects[i].iouValues[j];
					pos = j;
				}
			}
		}
		if (pos != 999) {
			updateTrackedObject(trackedObjects[i], detections[pos]);
		}
		else {
			if (trackedObjects[i].noNewDetection >= 7) {
				trackedObjects[i].ignore = 1;
			}
			else {
				calcNewPosition(trackedObjects[i]);
			}
		}
	}
}

void hungarian(int tries, std::vector<TrackedObject> &trackedObjects, std::vector<Detections> detections) {
	tries++;
	calcMatrix(trackedObjects, detections);
	for (int i = 0; i < trackedObjects.size(); i++) {
		int minNumber = 100;

		for (int j = 0; j < detections.size(); j++) {
			//change 100 to the number that you want to subtract as a maximum, so only detections that are close will be subtracted
			if (trackedObjects[i].iouValues[j] < 60 && trackedObjects[i].iouValues[j] < minNumber && trackedObjects[i].iouValues[j] != 0) {
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
			if (trackedObjects[i].iouValues[j] < 50) {
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
				if (trackedObjects[j].iouValues[position] < 50) {
					amount++;
				}
			}
		}

		if (amount == 1) {
			updateTrackedObject(trackedObjects[i], detections[position]);
		}

	}

	//Tracked Object cuts with no new detection
	for (int i = 0; i < trackedObjects.size(); i++) {
		int amount = 0;
		if (trackedObjects[i].updated == 1 || trackedObjects[i].ignore == 1) {
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
			if (trackedObjects[i].noNewDetection >= 7) {
				trackedObjects[i].ignore = 1;
			}
			else {

				calcNewPosition(trackedObjects[i]);
			}
		}
	}
	Mat values = calcMatrixo(trackedObjects, detections);

	assignNewDetection(trackedObjects, detections);
	recursiveAssigning(trackedObjects, detections);
	assignLastDetections(trackedObjects, detections);
}

int main() {
	const int dataset = 5;
	String filepath("data\\data_m4\\5\\");
	std::ostringstream seqinfoPath; seqinfoPath << "data\\data_m4\\" << dataset << "\\seqinfo.ini";
	int seqLength = GetPrivateProfileIntA("Sequence", "seqLength", 1050,  seqinfoPath.str().c_str());
	imWidth = GetPrivateProfileIntA("Sequence", "imWidth", 1920,  seqinfoPath.str().c_str());
	imHeight = GetPrivateProfileIntA("Sequence", "imHeight", 1080,  seqinfoPath.str().c_str());
	const int totalIDs = 150;
	const int detlimit = 10;

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
	const bool useMedianSizeFilter = true;
	float lower_p = 0.7, higher_p = 2;

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
		if (useMedianSizeFilter && det.size() >= detlimit) {
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
			}
		}

		for (int i = 0; i < det.size(); i++) {
			rectangle(imgCopy, det[i].getRect(), Scalar(255, 0, 0), 1);
		}

		if (pos >= 2) {
			hungarian( 0, trackedObjects, det);
		}

		lastFrame = in_img.clone();

		//draw tracked Objects
		for (int i = 0; i < trackedObjects.size(); i++) {
			if (trackedObjects[i].updated == 1 && trackedObjects[i].ignore == 0) {
				rectangle(in_img, trackedObjects[i].det.getRect(), Scalar(0, 0, 255), 1);
				cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,0,255 }, 2);
			}
			else if(trackedObjects[i].ignore == 0){
				rectangle(in_img, trackedObjects[i].det.getRect(), Scalar(0, 255, 0), 1);
				cv::putText(in_img, std::to_string(trackedObjects[i].id), { trackedObjects[i].getX(), trackedObjects[i].getY() + 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);
			}
			trackedObjects[i].updated = 0;
			trackedObjects[i].iouValues.clear();
			
		}

		//draw groundtruths
		/*
		for (int i = 1; i < nGroundtruths; i++) {
			rectangle(in_img, Rect(gt[i].left, gt[i].top, gt[i].width, gt[i].height), Scalar(255, 0, 0), 1);
		}*/
		
		std::cout << " Frame: " << pos << std::endl;
		//imshow("TrackedObjects", in_img);
		//imshow("Detections", imgCopy);
		// do 10 steps before waiting again 
		if (pos % 1 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}

		// write to output file
		for (int i = 0; i < trackedObjects.size(); i++) {
			if (trackedObjects[i].ignore == 1) {
				continue;
			}
			// <frame number>, tracked output... , <x>, <y>, <z>
			outputFile << pos << ' ' << trackedObjects[i].getOutput() << " -1 -1 -1" << std::endl;
		}

		cv::destroyAllWindows();
	}

	return 0;
}