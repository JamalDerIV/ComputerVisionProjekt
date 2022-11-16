#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;

/*
* global variable for the path to load the images from
*/
std::string globalImagePath()
{
	static std::string imagePath("data\\data_m2\\1\\");
	return imagePath;
}

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

public:
	/*
	* creates an background image out of x images
	*/
	void apply() {

	//load all images as grayscale into an array
	for (int i = 1; i < imagesToLoad+1; i++) {
		std::ostringstream img_name;
		char pos_str[7];
		sprintf_s(pos_str, "%0.6d", i);

		img_name << globalImagePath() << "input\\in" << pos_str << ".jpg";
		images[imagesToLoad-i] = cv::imread(img_name.str(), cv::IMREAD_GRAYSCALE);
	}

	//declares backgroundImage as a blank image the size xy of one the images in the array
	bgdImg = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1);

	//add the grayscales of all images by position row col and divide by images to load to get the average grayscale
	for (int row = 0; row < images[0].rows; row++) {
		for (int col = 0; col < images[0].cols; col++) {

			int average = 0;

			for (int j = 0; j < imagesToLoad;j++) {
				average += images[j].at<uchar>(row,col);
			}

			average /= imagesToLoad;

			// save the average grayscale from position row col into our backgroundImage
			bgdImg.at<uchar>(row, col) = average; 
		}
	}
}

	/*
	* subtracts the background from an image
	*/
	cv::Mat substraction(cv::Mat imgNew) {

		for (int row = 0; row < imgNew.rows; row++) {
			for (int col = 0; col < imgNew.cols; col++) {

				//substract the backround from our newly loaded image to only get the difference
				imgNew.at<uchar>(row, col) = abs(imgNew.at<uchar>(row, col) - bgdImg.at<uchar>(row, col));

				//paint everything white that has a grayscale over 45
				if (imgNew.at<uchar>(row, col) > 45) {
					imgNew.at<uchar>(row, col) = 255;
				}
				else {
					//paint black everything under grayscale 46
					imgNew.at<uchar>(row, col) = 0;
				}

			}
		}

		return imgNew;
	}
};


/* Evaluation 
*	used to calculate F-Sorce of a BG-Subtraction
 */
class Evaluation {
private:
	int tp = 0, // Changes/person detected and in GT
		fn = 0, // Changes/person not detected, but in GT
		fp = 0, // Changes/person detected, but not in  GT
		tn = 0; // Changes/person not detected and not in GT

public:
	/* 
	* Compares a mask and a given groundtruth image and 
	* add ups all true positves to false negatives pixels 
	*/
	void evaluate(Mat mask, Mat gt) {
		tp = 0; fn = 0; fp = 0; tn = 0;
		uchar whitePixel = 255;
		uchar blackPixel = 0;
		uchar gtPixel;
		uchar imagePixel;
		for (int row = 1; row < mask.rows; row++) {
			for (int col = 1; col < mask.cols; col++) {
				gtPixel = gt.at<uchar>(row, col);
				imagePixel = mask.at<uchar>(row, col);
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
				// Grayvalues are being ignored
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

	/*
	* Debug print to console to check local attributes 
	*/
	void printValues() {
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
	double fscore_own = 0,
		fscore_knn = 0,
		fscore_mog2 = 0;

	const int numberImages = 1200,
		startingPos = 300;
	int iterations = 0;

	bgSub image(50);
	image.apply();

	for (int pos = startingPos; pos <= numberImages; pos += 1) {
		// Loading Images
		std::ostringstream in_img_name, gt_img_name;
		char pos_str[7];
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name << globalImagePath() << "input\\in" << pos_str << ".jpg";
		gt_img_name << globalImagePath() << "groundtruth\\gt" << pos_str << ".png";

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

		// Appling BG Subtraction and evalutation
		mog2BS->apply(in_img, mog2Mask);
		mog2Eval.evaluate(mog2Mask, gt_img);
		if (mog2Eval.getFScore() >= 0)
			fscore_mog2 += mog2Eval.getFScore();

		knnBS->apply(in_img, knnMask);
		knnEval.evaluate(knnMask, gt_img);
		fscore_knn += knnEval.getFScore();

		in_img = image.substraction(in_img);
		ownEval.evaluate(in_img, gt_img);
		fscore_own += ownEval.getFScore();

		iterations++; 
		// putting text on image and show it
		//MOG 2
		putText(mog2Mask, "FScore: " + std::to_string(mog2Eval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("Mog2 Background Substraction", mog2Mask);

		//KNN
		putText(knnMask, "FScore: " + std::to_string(knnEval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("KNN Background Substraction", knnMask);

		// OWN
		putText(in_img, "FScore: " + std::to_string(ownEval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("Own Background Substraction", in_img);
		
		// do 10 steps before waiting again 
		if (pos % 100 == 0) {
			int wait = cv::waitKey(0);
			if (wait == 27) {
				break; // ESC Key
			}
		}
	}

	std::cout << "F Scores: " << std::endl;
	std::cout << "Mog2 : " << fscore_mog2/ iterations << std::endl;
	std::cout << "KNN  : " << fscore_knn / iterations << std::endl;
	std::cout << "Own  : " << fscore_own / iterations << std::endl;
	cv::destroyAllWindows();
	return 0;
}