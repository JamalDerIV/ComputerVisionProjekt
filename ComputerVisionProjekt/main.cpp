#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;

class bgSub {
private:
	int imagesToLoad;
	cv::Mat *images;
	cv::Mat bgdImg;

public:
	bgSub(int pos) {
		imagesToLoad = pos;
		images = new cv::Mat [pos];
	}

public:void apply() {

	//load all images into an array
	for (int i = 1; i < imagesToLoad+1; i++) {
		std::ostringstream img_name;
		char pos_str[7];
		sprintf_s(pos_str, "%0.6d", i);

		img_name << "data\\data_m2\\4\\input\\in" << pos_str << ".jpg";
		images[imagesToLoad-i] = cv::imread(img_name.str(), cv::IMREAD_GRAYSCALE);
	}

	//creates a blank image
	bgdImg = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1);

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
}
	  cv::Mat substraction(int pos) {

		  std::ostringstream new_img_name;
		  char pos_str[7];
		  sprintf_s(pos_str, "%0.6d", pos);
		  new_img_name << "data\\data_m2\\4\\input\\in" << pos_str << ".jpg";
		  cv::Mat imgNew = cv::imread(new_img_name.str(), cv::IMREAD_GRAYSCALE);

		  for (int row = 0; row < imgNew.rows; row++) {
			  for (int col = 0; col < imgNew.cols; col++) {

				  imgNew.at<uchar>(row, col) = abs(imgNew.at<uchar>(row, col) - bgdImg.at<uchar>(row, col));

				  //paint everything white that has a grayscale over 45 and black everything else
				  if (imgNew.at<uchar>(row, col) > 45) {
					  imgNew.at<uchar>(row, col) = 255;
				  }
				  else {
					  imgNew.at<uchar>(row, col) = 0;
				  }

			  }
		  }

		  return imgNew;
	  }
};





int main() {

	bgSub image(50);

	image.apply();

	for (int pos = 60; pos <= 1200; pos += 20) { // Stepping 10 Frames at a time might conflict with Background Subtraction

		cv::Mat in_img = image.substraction(pos);

		imshow("testing", in_img);

		int wait = cv::waitKey(0);
		if (wait == 27) break; // ESC Key

		cv::destroyAllWindows();
	}
	return 0;
}