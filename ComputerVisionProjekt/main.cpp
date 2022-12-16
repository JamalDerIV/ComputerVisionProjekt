#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
cv::Mat src, srcGray;
cv::Mat harrisDST, myHarrisCOPY, mC;

std::vector<Point2f> pointFinished;

//GoodFeaturesToTrack parameters
int maxDistance = 60;
float freeParameterHD = 0.246f;
int blockSize = 200;
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

	//check if the corner is smaller than x
	for (int i = 0; i < srcGray.rows; i++){
		for (int j = 0; j < srcGray.cols; j++){
			if (mC.at<float>(i, j) > qualityLevel * myHarrisMaxVal){
				//add the coordiantes to a vector
				point0.push_back(Point2f(i, j));
				amount++;
			}
		}
	}

	cv::Mat dw = cv::Mat::ones(cv::Size(srcGray.cols, srcGray.rows), CV_8UC1);
	//check if the inner loops corner is within maxDistance and flag the smaller one
	if (!point0.empty()) {
		for (int k = 0; k < point0.size() - 1;k++) {
			if (dw.at<uchar>(point0[k].x, point0[k].y) != 0) {
				for (int l = k + 1; l < point0.size(); l++) {
					if (dw.at<uchar>(point0[k].x, point0[k].y) != 0) {
						if (abs(point0[k].x - point0[l].x) < maxDistance && abs(point0[k].y - point0[l].y) < maxDistance) {
							if (mC.at<float>(point0[k].x, point0[k].y) > mC.at<float>(point0[l].x, point0[l].y)) {
								dw.at<uchar>(point0[l].x, point0[l].y) = 0;
								dw.at<uchar>(point0[k].x, point0[k].y) = 1;
							}
							else {
								dw.at<uchar>(point0[k].x, point0[k].y) = 0;
								dw.at<uchar>(point0[l].x, point0[l].y) = 1;
							}
						}
					}
				}
			}
		}
	}

	/*for (int k = 0; k < point0.size(); k++) {
		if (dw.at<uchar>(point0[k].x, point0[k].y) != 0) {
			pointFinished.push_back(point0[k]);
		}
	}*/

	//save the biggest corner
	float biggestCorner = 0;
	int cornerX=0, cornerY=0;
	bool newCorner = false;
	for (int k = 0; k < point0.size(); k++) {
		if (isBigger(mC.at<float>(point0[k].x, point0[k].y), biggestCorner)) {
			newCorner = true;
			biggestCorner = mC.at<float>(point0[k].x, point0[k].y);
			cornerX = point0[k].x;
			cornerY = point0[k].y;
		}
	}

	//save the biggest corner in pointFinished
	if (newCorner) {
		pointFinished.push_back(Point2f(cornerY, cornerX));
	}
}

void cornerDetection(cv::Mat mat, cv::Mat harrisDST, cv::Mat mask) {
	int apertureSize = 3;

	//get the eigenValues
	cornerEigenValsAndVecs(mat, harrisDST, blockSize, apertureSize);

	//calculate fore every pixel a corner using eigenValues and the free parameter for the harris detector
	mC = Mat(srcGray.size(), CV_32FC1);
	for (int i = 0; i < srcGray.rows; i++) {
		for (int j = 0; j < srcGray.cols; j++) {
			if (mask.at<uchar>(i, j) != 0) {
				float lambda_1 = harrisDST.at<Vec6f>(i, j)[0];
				float lambda_2 = harrisDST.at<Vec6f>(i, j)[1];

				//für jeden pixel wird ein wert berechnet, der besagt ob ein corner vorhanden sein kann
				mC.at<float>(i, j) = lambda_1 * lambda_2 - freeParameterHD * ((lambda_1 + lambda_2) * (lambda_1 + lambda_2));
			}
			else {
				mC.at<float>(i, j) = 0;
			}
		}
	}

	minMaxLoc(mC, &myHarrisMinVal, &myHarrisMaxVal);

	myGoodFeaturesToTrackFunction();
}

//claculates the IoU
double iou(Rect boxGT, Rect boxPredicted) {
	int xA = std::max(boxGT.x, boxPredicted.x),
		xB = std::min(boxGT.x + boxGT.width, boxPredicted.x + boxPredicted.width),
		yA = std::max(boxGT.y, boxPredicted.y),
		yB = std::min(boxGT.y + boxGT.height, boxPredicted.y + boxPredicted.height);
	if (xA > xB) return 0.0;
	if (yA > yB) return 0.0;
	// intersection
	int iArea = (xB - xA + 1) * (yB - yA + 1);
	//union
	int areaA = (boxGT.width + 1) * (boxGT.height + 1);
	int areaB = (boxPredicted.width + 1) * (boxPredicted.height + 1);

	return (double)iArea / (double)(areaA + areaB - iArea);
}

int main() {

	cv::Ptr<BackgroundSubtractorMOG2> mog2BS = createBackgroundSubtractorMOG2();
	
	mog2BS->setShadowThreshold(0.86);
	mog2BS->setShadowValue(0);
	mog2BS->setNMixtures(2);
	mog2BS->setHistory(100);
	
	std::ostringstream first_img_name;
	String filepath( "data\\data_m3\\1\\");
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

	// Meanshift Parameters
	int width = 100;
	int height = 180;
	float scaleFactor = 1.2f;

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

		cv::Mat imgOut;
		//cv::GaussianBlur(inputImg, inputImg,cv::Size(9,9),3,3);
		
		mog2BS->apply(inputImg, mog2Mask, 0.005);
		cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));

		if(pointFinished.size() < 1 && !personLeft) {
			darkenMidMat(mog2Mask);

			finishedMask = abs(mog2Mask - prevMask);

			prevMask = mog2Mask.clone();
			
			cv::erode(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			cv::dilate(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

			namedWindow("finishedMask", WND_PROP_FULLSCREEN);
			setWindowProperty("finishedMask", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
			imshow("finishedMask", finishedMask);
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
			cornerDetection(prevStaticGrayImg, harrisDST, finishedMask);

			if (pointFinished.size() > 0) {
				if (pointFinished[0].y > mog2Mask.rows * 0.9) {
					yOffset = height / 2;
				}
				else if (pointFinished[0].y < mog2Mask.rows * 0.1) {
					yOffset = -height / 2;
				}

				if (pointFinished[0].x > mog2Mask.cols * 0.9) {
					xOffset = width / 2;
				}
				else if (pointFinished[0].x < mog2Mask.cols * 0.1) {
					xOffset = -width / 2;
				}
			}
		}
			//goodFeaturesToTrack(prevStaticGrayImg, pointFinished, 1, 0.00001, 35, finishedMask, 200, true, 0.24);
		//}

		Mat grayImg;
		cvtColor(inputImg, grayImg, COLOR_BGR2GRAY);
		std::vector<uchar> status;
		std::vector<float> err;
		if (pointFinished.size() >= 1) {
			pointCreated = true;
			if (pointFinished[0].x < 0 || pointFinished[0].y < 0 || pointFinished[0].x > inputImg.cols || pointFinished[0].y > inputImg.rows) {
				std::cout << "----------------- person out of image! ------------------" << std::endl;
				personLeft = true;
				pointFinished.clear();
			}
		}

		if (pointFinished.size() > 0) {
			//cv::blur(prevStaticGrayImg, prevStaticGrayImg, cv::Size(9, 9));
			cv::blur(grayImg, grayImg, cv::Size(5, 5));
			namedWindow("grayImg", WND_PROP_FULLSCREEN);
			setWindowProperty("grayImg", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
			imshow("grayImg", grayImg);
		}

		if(pos >= startingThreshold && pointFinished.size() >= 1 && !personLeft){
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), criteriaMaxCount, criteriaEpsilon);
			calcOpticalFlowPyrLK(prevStaticGrayImg, grayImg, pointFinished, p1, status, err, Size(sizeNumber,sizeNumber), maxLevel, criteria);
		
			// Mean Shift Mask
			cvtColor(inputImg, hsv, COLOR_BGR2HSV);
			calcBackProject(&hsv, 1, channels, roi_hist, backProjectionMask, range);
			
			// calculating an ROI around pointFinished
			trackFrame.x = pointFinished[0].x - (width * 0.5);
			trackFrame.y = pointFinished[0].y - (height);
			int x = pointFinished[0].x - (width * (scaleFactor / 2)) + xOffset;
			int y = pointFinished[0].y - (height * (scaleFactor / 2)) + yOffset;
			// checking for frame borders 
			x = (x < 0) ? 0 : x;
			y = (y < 0) ? 0 : y;
			int cutWidth = (width * scaleFactor + x >= mog2Mask.cols) ? (mog2Mask.cols - x): width*scaleFactor; 
			int cutHeight = (height * scaleFactor + y >= mog2Mask.rows) ? (mog2Mask.rows - y): height*scaleFactor;
			// cutting backProjectionMask and mog2Mask together in the given ROI
			Rect cuttingSize = cv::Rect(x, y, cutWidth, cutHeight);
			Mat cutPerson = mog2Mask(cuttingSize);
			Mat combinedMask = backProjectionMask(cuttingSize) | cutPerson;
			personMask = Mat::zeros(mog2Mask.size(), mog2Mask.type());
			combinedMask.copyTo(personMask(cuttingSize));
			rectangle(inputImg, Rect(x, y, cutWidth, cutHeight), 1, 1); // black box indicating ROI for mean shift
			
			meanShift(personMask, trackFrame, term_crit);
			rectangle(inputImg, trackFrame, 255, 1); // blue box around person
		}

		if (pos >= frame) {
			double eval = 0;
			
			// while file has lines -> read line, calculate evaluation and draw rectangle 
			if (gtfile.is_open()) {
				Rect gt = Rect(bb_left, bb_top, bb_width, bb_height);
				eval = iou(gt, trackFrame);
				//rectangle(inputImg, gt, Scalar(0, 255, 0), 1);
				gtfile >> frame >> id >> bb_left >> bb_top >> bb_width >> bb_height;
				if (!gtfile) gtfile.close();
			}
			
			// while either our box or the gt box is present we want to sum up total evaluation
			if (!personLeft || gtfile.is_open()) {
				std::cout << " Evaluation = " << eval << std::endl;
				evalSum += eval;
				evalIterations++;
			}
			else {
				std::cout << " Total Eval Score: " << evalSum / evalIterations << std::endl;
			}
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

		/* Kontrast verändern
		Mat new_image = Mat::zeros(in_img.size(), in_img.type());

		cv::Mat lab_img;
		cv::cvtColor(in_img, lab_img, COLOR_BGR2Lab);

		std::vector<cv::Mat> lab_planes(3);
		cv::split(lab_img, lab_planes);

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(4);
		cv::Mat dst;
		clahe->apply(lab_planes[0], dst);

		dst.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_img);

		cv::Mat image_clahe;
		cv::cvtColor(lab_img, image_clahe, COLOR_Lab2BGR);
		*/

		namedWindow("Mog2 Background Substraction", WND_PROP_FULLSCREEN);
		setWindowProperty("Mog2 Background Substraction", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
		imshow("Mog2 Background Substraction", inputImg);
		
		int keyboard = waitKey();
		if (keyboard == 'q' || keyboard == 27)
			break;

		prevStaticGrayImg = grayImg.clone();
		if (!personLeft) {
			pointFinished = good_new;
		}

		cv::destroyAllWindows();
	}
	return 0;
}