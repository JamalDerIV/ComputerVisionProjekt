#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
cv::Mat src, srcGray;
cv::Mat harrisDST, myHarrisCOPY, mC;

std::vector<Point2f> pointFinished;

int maxDistance = 60;

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

void myGoodFeaturesToTrackFunction()
{
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
			if (mC.at<float>(i, j) > 0.00001* myHarrisMaxVal){
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
	int blockSize = 200, apertureSize = 3;

	//get the eigenValues
	cornerEigenValsAndVecs(mat,harrisDST,blockSize,apertureSize);

	//calculate fore every pixel a corner using eigenValues and the free parameter for the harris detector
	mC = Mat(srcGray.size(), CV_32FC1);
	for (int i = 0; i < srcGray.rows; i++)
	{
		for (int j = 0; j < srcGray.cols; j++)
		{
			if (mask.at<uchar>(i, j) != 0) {
				float lambda_1 = harrisDST.at<Vec6f>(i, j)[0];
				float lambda_2 = harrisDST.at<Vec6f>(i, j)[1];

				//für jeden pixel wird ein wert berechnet, der besagt ob ein corner vorhanden sein kann
				mC.at<float>(i, j) = lambda_1 * lambda_2 - 0.246f * ((lambda_1 + lambda_2) * (lambda_1 + lambda_2));
			}
			else {
				mC.at<float>(i, j) = 0;
			}
		}
	}

	minMaxLoc(mC, &myHarrisMinVal, &myHarrisMaxVal);

	myGoodFeaturesToTrackFunction();
}

int main() {

	cv::Ptr<BackgroundSubtractorMOG2> mog2BS = createBackgroundSubtractorMOG2();
	
	mog2BS->setShadowThreshold(0.86);
	mog2BS->setShadowValue(0);
	mog2BS->setNMixtures(2);
	mog2BS->setHistory(100);
	
	std::ostringstream first_img_name;
	char pos_str[7];
	sprintf_s(pos_str, "%0.6d", 1);
	first_img_name << "data\\data_m3\\2\\img1\\" << pos_str << ".jpg";

	cv::Mat first_img = cv::imread(first_img_name.str(), cv::IMREAD_COLOR);
	if (first_img.empty())
	{
		std::cout << "Could not read the image: " <<  first_img_name.str() << std::endl;
		return 1;
	}

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

	bool personLeft = false;
	
	int pos = 1;
	const int startingThreshold = pos + 35;
	while ( pos < 795){

		std::cout << pos++ << std::endl;

		std::ostringstream inImgName;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		inImgName <<  "data\\data_m3\\2\\img1\\"<< pos_str <<".jpg";
		
		cv::Mat inputImg = cv::imread(inImgName.str(), cv::IMREAD_COLOR);
		if (inputImg.empty())
		{
			std::cout << "Could not read the image: " << inImgName.str() << std::endl;
			return 1;
		}

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

		if (pointFinished.size() < 1 && pos >= startingThreshold && !personLeft) {
			prevStaticGrayImg = inputImg.clone();
			cvtColor(prevStaticGrayImg, prevStaticGrayImg, COLOR_BGR2GRAY);
			cornerDetection(prevStaticGrayImg, harrisDST, finishedMask);
		}
			//goodFeaturesToTrack(prevStaticGrayImg, pointFinished, 1, 0.00001, 35, finishedMask, 200, true, 0.24);
		//}

		Mat grayImg;
		cvtColor(inputImg, grayImg, COLOR_BGR2GRAY);
		// calculate optical flow
		std::vector<uchar> status;
		std::vector<float> err;
		if (pointFinished.size() >= 1) {
			if (pointFinished[0].x < 0 || pointFinished[0].y < 0) {
				std::cout << "----------------- person out of image! ------------------";
				personLeft = true;
				pointFinished.clear();
			}
		}

		if(pos >= startingThreshold && pointFinished.size() >= 1 && !personLeft){
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 5, 0.001);
			calcOpticalFlowPyrLK(prevStaticGrayImg, grayImg, pointFinished, p1, status, err, Size(14, 14), 5, criteria);
		}

		std::vector<Point2f> good_new;
		for (uint i = 0; i < pointFinished.size(); i++) {
			std::cout << " \t\ti: " << i << " \tstatus:" << status[i] << " \tpointFinished:" << pointFinished[i] << " \tp1:" << p1[i] << std::endl;
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
		pointFinished = good_new;

		cv::destroyAllWindows();
	}
	return 0;
}