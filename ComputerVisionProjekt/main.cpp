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

		img_name << "data\\data_m2\\1\\input\\in" << pos_str << ".jpg";
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
	  cv::Mat substraction(cv::Mat imgNew) {

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

class Evaluation {
private:
	int tp = 0, // Änderung/Person erkannt und ist im GT
		fn = 0, // Änderung/Person nicht erkannt, aber im GT
		fp = 0, // Änderung/Person erkannt, aber nicht im GT
		tn = 0; // Änderung/Person nicht erkannt und ist nicht im GT

public:
	void evaluate(Mat image, Mat gt) {
		tp = 0; fn = 0; fp = 0; tn = 0;
		uchar whitePixel = 255;
		uchar blackPixel = 0;
		uchar gtPixel;
		uchar imagePixel;
		for (int row = 1; row < image.rows; row++) {
			for (int col = 1; col < image.cols; col++) {
				gtPixel = gt.at<uchar>(row, col);
				imagePixel = image.at<uchar>(row, col);
				if (imagePixel == blackPixel && gtPixel == blackPixel) {
					tn++;
					continue;
				}
				if (imagePixel == whitePixel && gtPixel == whitePixel) {
					tp++;
					continue;
				}
				if (imagePixel == blackPixel && gtPixel == whitePixel) {
					fn++;
					continue;
				}
				if (imagePixel == whitePixel && gtPixel == blackPixel) {
					fp++;
					continue;
				}
				// Grauwerte werden ignoriert
			}
		}
	}

	float getAccuracy() {
		return (float)(tp + tn) / (tp + tn + fp + fn);
	}

	float getSpecifity() {
		return (float)tn / (tn + fp);
	}

	float getPrecision() {
		return (float)tp / (tp + fp);
	}

	float getRecall() { // Sensitivity
		return (float)tp / (tp + fn);
	}

	float getFScore() {
		float p = getPrecision();
		float r = getRecall();
		return  (2 * p * r) / (p + r);
	}

	void printValus() {
		std::cout << "tp: " << tp
			<< "   tn: " << tn
			<< "   fp: " << fp
			<< "   fn: " << fn
			<< "   sum: " << tp + tn + fp + fn
			<< "   total pix count: " << 720 * 576
			<< std::endl;
	}

};

int main() {
	Evaluation mog2Eval;
	Evaluation knnEval;
	Evaluation ownEval;

	cv::Ptr<BackgroundSubtractor> mog2BS = createBackgroundSubtractorMOG2();
	cv::Ptr<BackgroundSubtractor> knnBS = createBackgroundSubtractorKNN();

	cv::Mat mog2Mask, knnMask;

	bgSub image(50);
	image.apply();

	for (int pos = 300; pos <= 1200; pos += 1) {
		std::ostringstream in_img_name, gt_img_name;
		char pos_str[7];
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name << "data\\data_m2\\1\\input\\in" << pos_str << ".jpg";
		gt_img_name << "data\\data_m2\\1\\groundtruth\\gt" << pos_str << ".png";

		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_GRAYSCALE);
		cv::Mat gt_img = cv::imread(gt_img_name.str(), cv::IMREAD_GRAYSCALE);

		if (in_img.empty())
		{
			std::cout << "Could not read the image: " << in_img_name.str() << std::endl;
			return 1;
		}
		if (gt_img.empty())
		{
			std::cout << "Could not read the image: " << gt_img_name.str() << std::endl;
			return 1;
		}

		mog2BS->apply(in_img, mog2Mask);
		mog2Eval.evaluate(mog2Mask, gt_img);

		knnBS->apply(in_img, knnMask);
		knnEval.evaluate(knnMask, gt_img);

		in_img = image.substraction(in_img);
		ownEval.evaluate(in_img, gt_img);

		//MOG 2
		putText(mog2Mask, "FScore: " + std::to_string(mog2Eval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("Mog2 Background Substraction", mog2Mask);

		//KNN
		putText(knnMask, "FScore: " + std::to_string(knnEval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("KNN Background Substraction", knnMask);

		// OWN
		putText(in_img, "FScore: " + std::to_string(ownEval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("Own Background Substraction", in_img);
		
		if (pos % 10 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}
	}
	cv::destroyAllWindows();
	return 0;
}