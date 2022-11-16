#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;

cv::Mat bgSub(cv::Mat img1, cv::Mat img2) {
	cv::Mat solution;
	solution.create(img1.rows, img1.cols, CV_8UC3);
	for (int row = 0; row < solution.rows; row++) {
		for (int col = 0; col < solution.cols; col++) {
			cv::Vec3b whitePixel = { 255, 255, 255 };
			solution.at<cv::Vec3b>(row, col) = img1.at<cv::Vec3b>(row, col) - (whitePixel - img2.at<cv::Vec3b>(row, col));
		}
	}
	return solution;
}

class Evaluation {
private: 
	int tp=0, // Änderung/Person erkannt und ist im GT
		fn=0, // Änderung/Person nicht erkannt, aber im GT
		fp=0, // Änderung/Person erkannt, aber nicht im GT
		tn=0; // Änderung/Person nicht erkannt und ist nicht im GT

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
		return (float) (tp + tn) / (tp + tn + fp + fn);
	}

	float getSpecifity() {
		return (float) tn / (tn + fp);
	}

	float getPrecision() {
		return (float) tp / (tp + fp);
	}

	float getRecall() { // Sensitivity
		return (float) tp / (tp + fn);
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
			<< "   sum: " << tp+tn+fp+fn
			<< "   total pix count: " << 720*576
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

	for (int pos = 300; pos <= 1200; pos += 1) { // Stepping 10 Frames at a time might conflict with Background Subtraction
		std::ostringstream in_img_name, gt_img_name;
		char pos_str[7]; 
		sprintf_s(pos_str, "%0.6d", pos);
		in_img_name <<  "data\\data_m2\\1\\input\\in"<< pos_str <<".jpg";
		gt_img_name <<  "data\\data_m2\\1\\groundtruth\\gt"<< pos_str <<".png";
		
		cv::Mat in_img = cv::imread(in_img_name.str(), cv::IMREAD_COLOR);
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
		mog2Eval.printValus();

		knnBS->apply(in_img, knnMask);
		knnEval.evaluate(knnMask, gt_img);
		knnEval.printValus();

		ownEval.evaluate(gt_img, gt_img);
		ownEval.printValus();
		std::cout << "True F Score: " << ownEval.getFScore() << std::endl;
		//imshow("Own Substraction",  bgSub(in_img, gt_img));
		//imshow("Double Substraction",  in_img - (in_img - gt_img));
		
		//MOG 2
		putText(mog2Mask, "FScore: " + std::to_string(mog2Eval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, {100, 100, 100});
		imshow("Mog2 Background Substraction",  mog2Mask);
		//KNN
		putText(knnMask, "FScore: " + std::to_string(knnEval.getFScore()), Point(5, 30), FONT_HERSHEY_DUPLEX, 0.8, { 100, 100, 100 });
		imshow("KNN Background Substraction",  knnMask);

		int wait = cv::waitKey(0);
		if (wait == 27) break; // ESC Key

		cv::destroyAllWindows();
	}
	return 0;
}