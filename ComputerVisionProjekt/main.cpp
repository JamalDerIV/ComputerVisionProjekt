#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;

class bgSub {
private:
	int imagesToLoad;
	cv::Mat *images;

public:
	bgSub(int pos) {
		imagesToLoad = pos;
		images = new cv::Mat [pos];
	}

public:cv::Mat apply(int pos) {

	

	//load all images into an array
	for (int i = 1; i < imagesToLoad+1; i++) {
		std::ostringstream img_name;
		char pos_str[7];
		sprintf_s(pos_str, "%0.6d", i);

		img_name << "data\\data_m2\\1\\input\\in" << pos_str << ".jpg";
		images[imagesToLoad-i] = cv::imread(img_name.str(), cv::IMREAD_GRAYSCALE);
	}

	//create a blank image
	cv::Mat bgdImg(cv::Size(images[0].cols, images[0].rows), CV_8UC1);

	//add the image grayscales and divide by images to load to get the average grayscale
	for (int row = 0; row < images[0].rows; row++) {
		for (int col = 0; col < images[0].cols; col++) {

			int average = 0;

			for (int j = 0; j < imagesToLoad;j++) {
				average += images[j].at<uchar>(row,col);
			}

			average /= imagesToLoad;
			bgdImg.at<uchar>(row, col) = average;
		}
	}
	
	std::ostringstream new_img_name;
	char pos_str[7];
	sprintf_s(pos_str, "%0.6d", pos);
	new_img_name << "data\\data_m2\\1\\input\\in" << pos_str << ".jpg";
	cv::Mat imgNew = cv::imread(new_img_name.str(), cv::IMREAD_GRAYSCALE);

	imshow("tei", imgNew);

	for (int row = 0; row < imgNew.rows; row++) {
		for (int col = 0; col < imgNew.cols; col++) {

			imgNew.at<uchar>(row, col) = abs(imgNew.at<uchar>(row, col) - bgdImg.at<uchar>(row, col));

			if (imgNew.at<uchar>(row, col) > 45) {
				imgNew.at<uchar>(row, col) = 255;
			}
			else {
				imgNew.at<uchar>(row, col) = 0;
			}

		}
	}

	imshow("testi", bgdImg);

	/*cv::Mat graymat;
	cv::cvtColor(solution, graymat, cv::COLOR_BGR2GRAY);


	for (int row = 0; row < graymat.rows; row++) {
		for (int col = 0; col < graymat.cols; col++) {
			cv::Vec3b ignore = { 20,20,20 };
			cv::Vec3b whitePixel = { 255,255,255 };
			cv::Vec3b blackPixel = { 0,0,0 };

			cv::Vec3b color = graymat.at<uchar>(row, col);

			//std::cout << std::to_string(color[0]> ignore[0]);
			//std::cout << std::to_string(graymat.at<cv::Vec3b>(row, col)[2]);

			if (ignore[0] > graymat.at<uchar>(row, col)) {
				graymat.at<uchar>(row, col) = 0;
			}
			else {
				graymat.at<uchar>(row, col) = 255;
			}
		}
	}*/

	return imgNew;
}
};





int main() {
	for (int pos = 60; pos <= 1200; pos += 20) { // Stepping 10 Frames at a time might conflict with Background Subtraction

		bgSub image(50);
		
		cv::Mat in_img = image.apply(pos);

		imshow("testing", in_img);

		int wait = cv::waitKey(0);
		if (wait == 27) break; // ESC Key

		cv::destroyAllWindows();
	}
	return 0;
}