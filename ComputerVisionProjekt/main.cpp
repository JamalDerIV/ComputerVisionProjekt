#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;


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

	
	int pos = 11;
	const int startingThreshold = pos + 10;
	while ( pos < 1050){

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

		if(pos <= startingThreshold) {
			mog2BS->apply(inputImg, mog2Mask, 0.005); 
			cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			finishedMask = abs(mog2Mask - prevMask);
			cv::erode(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));
			cv::dilate(finishedMask, finishedMask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			prevMask = mog2Mask.clone();
			imshow("aa", finishedMask);
		}
		
		if (pos == startingThreshold) {
			prevGrayImg = inputImg.clone();
			cvtColor(prevGrayImg, prevGrayImg, COLOR_BGR2GRAY);
			goodFeaturesToTrack(prevGrayImg, p0, 200, 0.0001, 10, finishedMask, 25, false, 0.04);
		}
		
		Mat grayImg;
		cvtColor(inputImg, grayImg, COLOR_BGR2GRAY);
		// calculate optical flow
		std::vector<uchar> status;
		std::vector<float> err;
		if(pos >= startingThreshold){
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
			calcOpticalFlowPyrLK(prevGrayImg, grayImg, p0, p1, status, err, Size(15, 15), 2, criteria);
		}

		std::vector<Point2f> good_new;
		for (uint i = 0; i < p0.size(); i++) {
			std::cout << " \t\ti: " << i << " \tstatus:" << status[i] << " \tp0:" << p0[i] << " \tp1:" << p1[i] << std::endl;
			// Select good points
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				line(drawingMask, p1[i], p0[i], colors[i], 2);
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
		
		//cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
		//cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

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