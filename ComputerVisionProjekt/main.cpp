#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;

void darkenMidMat(cv::Mat mat) {
	for (int row = 15; row < mat.rows-15; row++) {
		for (int col = 15; col < mat.cols-15; col++) {
			mat.at<uchar>(row, col) = 0;
		}
	}
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
	first_img_name << "data\\data_m3\\1\\img1\\" << pos_str << ".jpg";

	cv::Mat first_img = cv::imread(first_img_name.str(), cv::IMREAD_COLOR);
	if (first_img.empty())
	{
		std::cout << "Could not read the image: " <<  first_img_name.str() << std::endl;
		return 1;
	}

	// Meanshift
	int width = 100;
	int height = 180;
	float scaleFactor = 1.2f;

	Rect trackFrame(0, 0, width, height);
	Mat roi, hsv_roi, mask;
	// set up the ROI for tracking // Region of Interest 
	roi = first_img(trackFrame);
	cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
	//inRange(hsv_roi, Scalar(90, 37, 100), Scalar(110, 120, 180), mask);
	Scalar lowb = Scalar(50, 20, 50);
	Scalar highb = Scalar(100, 120, 200);
	inRange(hsv_roi, lowb, highb, mask);
	Mat personMask;
	int yOffset = 0, xOffset = 0;

	TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
	Mat hsv;
	Mat dst = Mat::zeros(first_img.size(), first_img.type());

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
	Mat prevGrayImg;
	std::vector<Point2f> p0, p1;

	// Take first frame and find corners in it
	// Create a mask image for drawing purposes
	Mat drawingMask = Mat::zeros(first_img.size(), first_img.type());
	cv::Mat mog2Mask, prevMask, finishedMask;
	prevMask = cv::Mat::zeros(cv::Size(first_img.cols, first_img.rows), CV_8UC1);

	bool personLeft = false;
	
	int pos = 1;
	const int startingThreshold = pos + 5;
	while ( pos < 795){

		std::cout << pos++ << std::endl;

		std::ostringstream inImgName;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		inImgName <<  "data\\data_m3\\1\\img1\\"<< pos_str <<".jpg";
		
		cv::Mat inputImg = cv::imread(inImgName.str(), cv::IMREAD_COLOR);
		if (inputImg.empty())
		{
			std::cout << "Could not read the image: " << inImgName.str() << std::endl;
			return 1;
		}


		mog2BS->apply(inputImg, mog2Mask, 0.005);
		cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
		if(p0.size() < 1 && !personLeft) {
			// 
			darkenMidMat(mog2Mask);

			finishedMask = abs(mog2Mask - prevMask);

			prevMask = mog2Mask.clone();
			
			cv::erode(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			cv::dilate(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
			namedWindow("aa", WND_PROP_FULLSCREEN);
			setWindowProperty("aa", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
			imshow("aa", finishedMask);
		}

		if (p0.size() < 1 && pos >= startingThreshold && !personLeft) {
			prevGrayImg = inputImg.clone();
			cvtColor(prevGrayImg, prevGrayImg, COLOR_BGR2GRAY);
			goodFeaturesToTrack(prevGrayImg, p0, 1, 0.00001, 35, finishedMask, 200, true, 0.24);

			if (p0.size() > 0) {
				if (p0[0].y > mog2Mask.rows * 0.9) {
					yOffset = height / 2;
				}
				else if (p0[0].y < mog2Mask.rows * 0.1) {
					yOffset = -height / 2;
				}

				if (p0[0].x > mog2Mask.cols * 0.9) {
					xOffset = width / 2;
				}
				else if (p0[0].x < mog2Mask.cols * 0.1) {
					xOffset = -width / 2;
				}
			}
			
		}

		Mat grayImg;
		cvtColor(inputImg, grayImg, COLOR_BGR2GRAY);
		std::vector<uchar> status;
		std::vector<float> err;
		if (p0.size() >= 1) {
			if (p0[0].x < 0 || p0[0].y < 0) {
				std::cout << "----------------- person out of image! ------------------";
				personLeft = true;
				p0.clear();
			}
		}

		if(pos >= startingThreshold && p0.size() >= 1 && !personLeft){
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 5, 0.001);
			calcOpticalFlowPyrLK(prevGrayImg, grayImg, p0, p1, status, err, Size(14, 14), 5, criteria);

			// Mean Shift Mask
			cvtColor(inputImg, hsv, COLOR_BGR2HSV);
			calcBackProject(&hsv, 1, channels, roi_hist, dst, range);
			
			trackFrame.x = p0[0].x - (width * 0.5);
			trackFrame.y = p0[0].y - (height);
			int x = p0[0].x - (width * (scaleFactor / 2)) + xOffset;
			x = (x < 0) ? 0 : x;
			int y = p0[0].y - (height * (scaleFactor / 2)) + yOffset;
			y = (y < 0) ? 0 : y;
			int cutWidth = (width * scaleFactor + x >= mog2Mask.cols) ? (mog2Mask.cols - x): width*scaleFactor; 
			int cutHeight = (height * scaleFactor + y >= mog2Mask.rows) ? (mog2Mask.rows - y): height*scaleFactor;
			Rect cuttingSize = cv::Rect(x, y, cutWidth, cutHeight);
			Mat cutPerson = mog2Mask(cuttingSize);
			Mat combinedMask = dst(cuttingSize) | cutPerson;
			personMask = Mat::zeros(mog2Mask.size(), mog2Mask.type());
			combinedMask.copyTo(personMask(cuttingSize));
			rectangle(inputImg, Rect(x, y, cutWidth, cutHeight), 1, 1);
			
			meanShift(personMask, trackFrame, term_crit);
			rectangle(inputImg, trackFrame, 255, 1);
		}

		std::vector<Point2f> good_new;
		for (uint i = 0; i < p0.size(); i++) {
			// Select good points
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				line(drawingMask, p1[i], p0[i], colors[i], 2);
				circle(inputImg, p1[i], 5, colors[i], -1);
			}
		}
		add(inputImg, drawingMask, inputImg);

		/* Kontrast ver�ndern
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
		
		//cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
		//cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

		namedWindow("Mog2 Background Substraction", WND_PROP_FULLSCREEN);
		setWindowProperty("Mog2 Background Substraction", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

		imshow("Mog2 Background Substraction", inputImg);
		
		int keyboard = waitKey();
		if (keyboard == 'q' || keyboard == 27)
			break;

		prevGrayImg = grayImg.clone();
		p0 = good_new;

		cv::destroyAllWindows();
	}
	return 0;
}