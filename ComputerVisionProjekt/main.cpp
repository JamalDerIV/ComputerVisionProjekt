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

	std::ostringstream old_img_name;
	char pos_str[7];
	sprintf_s(pos_str, "%0.6d", 150);
	old_img_name << "data\\data_m3\\1\\img1\\" << pos_str << ".jpg";

	cv::Mat old_img = cv::imread(old_img_name.str(), cv::IMREAD_COLOR);
	if (old_img.empty())
	{
		std::cout << "Could not read the image: " <<  old_img_name.str() << std::endl;
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
	Mat old_frame, old_gray;
	std::vector<Point2f> p0, p1;

	// Take first frame and find corners in it
	old_frame = old_img.clone();
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
	goodFeaturesToTrack(old_gray, p0, 300, 0.3, 7, Mat(), 25, false, 0.04);
	// Create a mask image for drawing purposes
	Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

	int pos = 151;
	while ( pos < 1050){
		std::cout << pos++ << std::endl;

		std::ostringstream in_img_name, gt_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  "data\\data_m3\\1\\img1\\"<< pos_str <<".jpg";
		
		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);
		if (in_img.empty())
		{
			std::cout << "Could not read the image: " << in_img_name.str() << std::endl;
			return 1;
		}
		
		cv::Mat mog2Mask;
		//cv::GaussianBlur(in_img, in_img, cv::Size(3, 3), 0.2, 0.2, 4);
		mog2BS->apply(in_img, mog2Mask, 0.005);
		


		Mat in_img_gray;
		cvtColor(in_img, in_img_gray, COLOR_BGR2GRAY);
		// calculate optical flow
		std::vector<uchar> status;
		std::vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(old_gray, in_img_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
		std::vector<Point2f> good_new;
		for (uint i = 0; i < p0.size(); i++)
		{
			// Select good points
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				line(mask, p1[i], p0[i], colors[i], 2);
				circle(in_img, p1[i], 5, colors[i], -1);
			}
		}
		Mat img;
		add(in_img, mask, img);	
		
		cv::erode(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
		cv::dilate(mog2Mask, mog2Mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

		imshow("Mog2 Background Substraction", img);
		
		int keyboard = waitKey();
		if (keyboard == 'q' || keyboard == 27)
			break;

		old_gray = in_img_gray.clone();
		p0 = good_new;

		cv::destroyAllWindows();
	}
	return 0;
}