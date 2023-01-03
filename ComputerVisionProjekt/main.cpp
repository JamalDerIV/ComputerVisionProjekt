#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
cv::Mat src, srcGray;
cv::Mat harrisDST, myHarrisCOPY, myCorners;

std::vector<Point2f> pointFinished;

//GoodFeaturesToTrack parameters
int maxDistance = 60;
float freeParameterHD = 0.206f;
int blockSize = 100;
float qualityLevel = 0.00001;

//Termcriteria parameters to terminate the search after the specific max count
int criteriaMaxCount = 5;
int criteriaEpsilon = 0.001;

//CalcopticalFlow parameters
int maxLevel = 5;
int sizeNumber = 14;


RNG rng(12345);

double myHarrisMinVal, myHarrisMaxVal;

void darkenMidMat(cv::Mat mat) {
	for (int row = 15; row < mat.rows-15; row++) {
		for (int col = 15; col < mat.cols-15; col++) {
			mat.at<uchar>(row, col) = 0;
		}
	}
}

bool isBigger(float a, float b) {
	return a > b;
}

void myGoodFeaturesToTrackFunction(){
	myHarrisCOPY = src.clone();

	int multi = 1;
	int nms = 3*multi;
	int amount = 0;
	int adNumber = 0;
	if (multi == 1) {
		adNumber = 1;
	}
	std::vector<Point2f> point0;
	std::vector<float> corners;
	int iMax, jMax, m, n, o, p;
	/*
	* calculate the maximum row and cols our non maximum suppression can go, so we can still check the left out ones in the following for loops
	iMax = srcGray.rows - (srcGray.rows % ((nms / 2) + (nms / 3) + adNumber));
	jMax = srcGray.cols - (srcGray.cols % ((nms / 2) + (nms / 3) + adNumber));

	// checking in a ixi block of the matrix if there are more than one corner and only keeps the bggest corner
	for (int i = 0; i < iMax; i += ((nms / 2) + (nms / 3)+adNumber)) {
		for (int j = 0; j < jMax; j += ((nms / 2) + (nms / 3)+adNumber)) {
			float tempNumber = 0;
			int tempX = 0, tempY = 0;
			for (m = i; m < i + ((nms / 2) + (nms / 3) + adNumber); m++) {
				for (n = j; n < j + ((nms / 2) + (nms / 3) + adNumber); n++) {
					float tmpNow = mC.at<float>(m, n);
					if (m == i - nms / 3 && n == j - nms / 3) {
						tempNumber = tmpNow;
						tempX = m;
						tempY = n;
					}
					if (isBigger(tempNumber, tmpNow)) {
						mC.at<float>(m, n) = float(0);
					}
					else {
						tempNumber = mC.at<float>(m, n);
						mC.at<float>(tempX, tempY) = 0;
						tempX = m;
						tempY = n;
					}
				}
			}
		}
	}

	//checks the (if) missed out bottom last pixels by ixi blocks to also minimize the corner amount 
	for (o = 0; o < jMax; o += ((nms / 2) + (nms / 3) + adNumber)) {
		float tempNumber = 0;
		int tempX = 0, tempY = 0;
		for (int k = iMax; k < srcGray.rows; k++) {
			for (int l = o; l < o + ((nms / 2) + (nms / 3) + adNumber); l++) {
				float tmpNow = mC.at<float>(k, l);
				if (k == iMax && l == o) {
					tempNumber = tmpNow;
					tempX = k;
					tempY = l;
				}
				if (isBigger(tempNumber, tmpNow)) {
					mC.at<float>(k, l) = float(0);
				}
				else {
					tempNumber = mC.at<float>(k, l);
					mC.at<float>(tempX, tempY) = 0;
					tempX = k;
					tempY = l;
				}
			}
		}
	}

	//checks the (if) missed out right side last pixels by ixi blocks to also minimize the corner amount
	for (p = 0; p < iMax; p += ((nms / 2) + (nms / 3) + adNumber)) {
		float tempNumber = 0;
		int tempX = 0, tempY = 0;
		for (int k = p; k < p + ((nms / 2) + (nms / 3) + adNumber); k++) {
			for (int l = jMax; l < srcGray.cols; l++) {
				float tmpNow = mC.at<float>(k, l);
				if (k == p && l == jMax) {
					tempNumber = tmpNow;
					tempX = k;
					tempY = l;
				}
				if (isBigger(tempNumber, tmpNow)) {
					mC.at<float>(k, l) = float(0);
				}
				else {
					tempNumber = mC.at<float>(k, l);
					mC.at<float>(tempX, tempY) = 0;
					tempX = k;
					tempY = l;
				}
			}
		}
	}

	//checks the (if) missed out bottom right side last pixels by ixi blocks to also minimize the corner amount
	float tempNumber = 0;
	int tempX = 0, tempY = 0;
	for (int k = iMax; k < srcGray.rows; k++) {
		for (int l = jMax; l < srcGray.cols; l++) {
			float tmpNow = mC.at<float>(k, l);
			if (k == iMax && l == jMax) {
				tempNumber = tmpNow;
				tempX = k;
				tempY = l;
			}
			if (isBigger(tempNumber, tmpNow)) {
				mC.at<float>(k, l) = float(0);
			}
			else {
				tempNumber = mC.at<float>(k, l);
				mC.at<float>(tempX, tempY) = 0;
				tempX = k;
				tempY = l;
			}
		}
	}*/

	//check if the corner is smaller than qualityLevel * myHarrisMaxVal which is the global maximum from the arrray
	for (int i = 0; i < srcGray.rows; i++){
		for (int j = 0; j < srcGray.cols; j++){
			if (myCorners.at<float>(i, j) > qualityLevel * myHarrisMaxVal){
				//add the coordiantes to a vector
				point0.push_back(Point2f(i, j));
				amount++;
			}
		}
	}

	//creates a matrix full of ones in the size of srcGray
	cv::Mat flagMatrix = cv::Mat::ones(cv::Size(srcGray.cols, srcGray.rows), CV_8UC1);

	//check if the inner loops corner is within maxDistance and flags the smaller one with 0
	if (!point0.empty()) {
		for (int k = 0; k < point0.size() - 1;k++) {
			if (flagMatrix.at<uchar>(point0[k].x, point0[k].y) != 0) {
				for (int l = k + 1; l < point0.size(); l++) {
					if (flagMatrix.at<uchar>(point0[k].x, point0[k].y) != 0) {
						if (abs(point0[k].x - point0[l].x) < maxDistance && abs(point0[k].y - point0[l].y) < maxDistance) {
							if (myCorners.at<float>(point0[k].x, point0[k].y) > myCorners.at<float>(point0[l].x, point0[l].y)) {
								flagMatrix.at<uchar>(point0[l].x, point0[l].y) = 0;
								flagMatrix.at<uchar>(point0[k].x, point0[k].y) = 1;
							}
							else {
								flagMatrix.at<uchar>(point0[k].x, point0[k].y) = 0;
								flagMatrix.at<uchar>(point0[l].x, point0[l].y) = 1;
							}
						}
					}
				}
			}
		}
	}

	//saves all the nonflagged corner in pointFinished
	for (int k = 0; k < point0.size(); k++) {
		if (flagMatrix.at<uchar>(point0[k].x, point0[k].y) != 0) {
			pointFinished.push_back(point0[k]);
		}
	}
	/*
	//save the biggest corner of the array
	float biggestCorner = 0;
	int cornerX=0, cornerY=0;
	bool newCorner = false;
	for (int k = 0; k < point0.size(); k++) {
		if (isBigger(myCorners.at<float>(point0[k].x, point0[k].y), biggestCorner)) {
			newCorner = true;
			biggestCorner = myCorners.at<float>(point0[k].x, point0[k].y);
			cornerX = point0[k].x;
			cornerY = point0[k].y;
		}
	}

	//save the biggest corner in pointFinished
	if (newCorner) {
		pointFinished.push_back(Point2f(cornerY, cornerX));
	}*/
}

/* function to calculate a corner for every pixel in source image*/
void cornerDetection(cv::Mat mat, cv::Mat harrisDST, cv::Mat mask) {
	int apertureSize = 3;

	//get the eigenValues for the corner calculation
	cornerEigenValsAndVecs(mat, harrisDST, blockSize, apertureSize);

	//calculate fore every pixel a corner using eigenValues and the free parameter for the harris detector
	myCorners = Mat(srcGray.size(), CV_32FC1);
	for (int i = 0; i < srcGray.rows; i++) {
		for (int j = 0; j < srcGray.cols; j++) {
			//use the mask to ignore corners and set the ignores ones to 0
			if (mask.at<uchar>(i, j) != 0) {
				float lambda_1 = harrisDST.at<Vec6f>(i, j)[0];
				float lambda_2 = harrisDST.at<Vec6f>(i, j)[1];

				//calculate the corner with the eigenvalues and the free parameter (same result as the harris corner detector)
				myCorners.at<float>(i, j) = lambda_1 * lambda_2 - freeParameterHD * ((lambda_1 + lambda_2) * (lambda_1 + lambda_2));
			}
			else {
				myCorners.at<float>(i, j) = 0;
			}
		}
	}

	//get the global minimum and maximum from the myCorners array
	minMaxLoc(myCorners, &myHarrisMinVal, &myHarrisMaxVal);

	myGoodFeaturesToTrackFunction();
}

int main() {
	cv::Ptr<BackgroundSubtractorMOG2> mog2BS = createBackgroundSubtractorMOG2();
	
	mog2BS->setShadowThreshold(0.86);
	mog2BS->setShadowValue(0);
	mog2BS->setNMixtures(2);
	mog2BS->setHistory(100);
	
	std::ostringstream first_img_name;
	String filepath( "data\\data_m4\\1\\");
	char pos_str[7];
	sprintf_s(pos_str, "%0.6d", 1);
	first_img_name << filepath <<"img1\\" << pos_str << ".jpg";

	cv::Mat first_img = cv::imread(first_img_name.str(), cv::IMREAD_COLOR);
	if (first_img.empty())
	{
		std::cout << "Could not read the image: " <<  first_img_name.str() << std::endl;
		return 1;
	}

	// Reading (custom) gt file
	//  -> the ',' seperator has been replaced with a ' ' whitespace
	std::ifstream gtfile(filepath + "\\gt\\gt_single_space.txt");
	int frame, id, bb_left, bb_top, bb_width, bb_height;
	gtfile >> frame >> id >> bb_left >> bb_top >> bb_width >> bb_height;
	double evalSum = 0;
	int evalIterations = 0;
	if (!gtfile.is_open()) {
		std::cout << "Could not open Ground Truth file. Evaluation will be skipped." << std::endl;
	}

	std::ofstream ownFile;
	ownFile.open("output\\ownEval.txt");

	// Meanshift Parameters
	int width = 135;
	int height = 380;
	float scaleFactor = 1.3f;

	Rect cuttingSize;
	Rect trackFrame(0, 0, width, height);
	Mat roi, hsv_roi, mask, personMask;
	int yOffset = 0, xOffset = 0;
	TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
	Mat hsv;
	Mat backProjectionMask = Mat::zeros(first_img.size(), first_img.type());
	// set up the ROI for tracking 
	roi = first_img(trackFrame);
	cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
	inRange(hsv_roi, Scalar(50, 20, 50), Scalar(100, 120, 200), mask);

	// calculate Histogram for mean shift
	float range_[] = { 0, 180 };
	const float* range[] = { range_ };
	int histSize[] = { 180 };
	int channels[] = { 0 };
	Mat roi_hist;
	calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
	normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);
	
	// Create some random colors
	std::vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 300; i++)
	{
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r, g, b));
	}
	Mat prevGrayImg,prevStaticGrayImg;
	std::vector<Point2f> p1;

	// Create a mask image for drawing purposes
	Mat drawingMask = Mat::zeros(first_img.size(), first_img.type());
	cv::Mat mog2Mask, prevMask, finishedMask;
	prevMask = cv::Mat::zeros(cv::Size(first_img.cols, first_img.rows), CV_8UC1);

	bool personLeft = false, pointCreated = false;
	
	int pos = 1;
	const int startingThreshold = pos + 5;
	while ( pos < 795){
		std::cout << "Frame: " << pos++ << std::endl;

		std::ostringstream inImgName;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		inImgName << filepath << "img1\\"<< pos_str <<".jpg";

		cv::Mat inputImg = cv::imread(inImgName.str(), cv::IMREAD_COLOR);
		if (inputImg.empty())
		{
			std::cout << "Could not read the image: " << inImgName.str() << std::endl;
			return 1;
		}
		
		mog2BS->apply(inputImg, mog2Mask);
		cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
		imshow("maske", mog2Mask);

		if(pointFinished.size() < 1 && !personLeft) {
			//darkenMidMat(mog2Mask);

			finishedMask = abs(mog2Mask - prevMask);

			prevMask = mog2Mask.clone();
			
			cv::erode(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			cv::dilate(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

		}

		prevGrayImg = inputImg.clone();
		cvtColor(prevGrayImg, prevGrayImg, COLOR_BGR2GRAY);

		srcGray = prevGrayImg;
		src = inputImg;

		if (pointCreated && pointFinished.size() < 1) {
			personLeft = true;
		}

		if (pointFinished.size() < 1 && pos >= startingThreshold && !personLeft) {
			prevStaticGrayImg = inputImg.clone();
			cvtColor(prevStaticGrayImg, prevStaticGrayImg, COLOR_BGR2GRAY);
			
			cornerDetection(prevStaticGrayImg, harrisDST, mog2Mask);

			//the official goodFeaturesToTrack function for comparing uses 
			//goodFeaturesToTrack(prevStaticGrayImg, pointFinished, 1, 0.00001, 35, finishedMask, 200, true, 0.24);

		}

		Mat grayImg;
		cvtColor(inputImg, grayImg, COLOR_BGR2GRAY);
		std::vector<uchar> status;
		std::vector<float> err;

		//set the bool personLeft to true if the optical flow leaves our frame
		if (pointFinished.size() >= 1) {
			pointCreated = true;
			if (pointFinished[0].x < 0 || pointFinished[0].y < 0 || pointFinished[0].x > inputImg.cols || pointFinished[0].y > inputImg.rows) {
				std::cout << "----------------- person out of image! ------------------" << std::endl;
				personLeft = true;
				pointFinished.clear();
			}
		}

		if (pointFinished.size() > 0) {
			//only used in sequences where the blur will have a better effect
			//cv::blur(grayImg, grayImg, cv::Size(5, 5));
		}

		if(pos >= startingThreshold && pointFinished.size() >= 1 && !personLeft){
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), criteriaMaxCount, criteriaEpsilon);
			calcOpticalFlowPyrLK(prevStaticGrayImg, grayImg, pointFinished, p1, status, err, Size(sizeNumber,sizeNumber), maxLevel, criteria);
		
		}

		if (pos >= frame) {
			double eval = 0;
		}

		std::vector<Point2f> good_new;
		for (uint i = 0; i < pointFinished.size(); i++) {
			// Select good points
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				line(drawingMask, p1[i], pointFinished[i], colors[i], 2);
				circle(inputImg, p1[i], 5, colors[i], -1);
			}
		}

		add(inputImg, drawingMask, inputImg);

		//std::string name = pos_str;
		//imwrite("inputImg\\" + name + ".jpg", inputImg);
		//if (!personMask.empty()) {
		//	imwrite("personMask\\" + name + ".jpg", personMask);
		//}

		imshow("Image",inputImg);
		
		int keyboard = waitKey();
		if (keyboard == 'q' || keyboard == 27)
			break;

		prevStaticGrayImg = grayImg.clone();
		if (!personLeft) {
			pointFinished = good_new;
		}

		cv::destroyAllWindows();
	}
	//ownFile << "\t" << evalSum / evalIterations << std::endl;
	return 0;
}