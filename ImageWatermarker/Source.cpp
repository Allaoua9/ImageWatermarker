#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

namespace watermark {

	int embedWatermark(std::vector<std::string> imagesPath, string watermarkPath, float alpha) {

		/* ----------------   Loading images  ---------------------- */
		int imgNumber = imagesPath.size();
		std::vector<Mat> originalImages;
		int i = 0;
		while (i < imgNumber) {
			Mat imageread = imread(imagesPath.at(i), IMREAD_COLOR);
			if (!imageread.data) {
				cout << "Could not open or find the image at : " << imagesPath.at(i) << std::endl;
				return -1;
			}
			originalImages.push_back(imageread);
			i++;
		}

		Mat watermark = imread(watermarkPath, IMREAD_GRAYSCALE);
		if (!watermark.data) {
			cout << "Could not open or find the watermark image at : " << watermarkPath << std::endl;
			return -1;
		}
		/* ------------------------------- Begining watermark embedding----------------------------- */

		// Preprocessing watermark
		// Applying Treshold
		cout << "Preprocessing watermark" << endl;
		threshold(watermark, watermark, 128, 255, THRESH_BINARY);

		// Converting Watermark image into array
		std::vector<uchar> watermarkArray;
		if (watermark.isContinuous()) {
			watermarkArray.assign(watermark.datastart, watermark.dataend);
		}
		else {
			for (int i = 0; i < watermark.rows; ++i) {
				watermarkArray.insert(watermarkArray.end(), watermark.ptr<uchar>(i), watermark.ptr<uchar>(i)+watermark.cols);
			}
		}
		// Splitting image into channels
		imgNumber = originalImages.size();
		cout << "Number of images : " << imgNumber << endl;
		
		int redundancy = 0;

		i = 0;
		// j represents index of current watermark pixel;
		int j = 0;
		// loop through all the images to insert the watermark
		while ( i < imgNumber) {
			Mat image = originalImages.at(i);
			// Spliting image into Three channels assuming it is an RGB image.
			Mat channels[3];
			split(image, channels);
			// Converting the blue channel of the image into float
			channels[0].convertTo(channels[0], CV_32FC1, 1.0);
			channels[1].convertTo(channels[1], CV_32FC1, 1.0);
			channels[2].convertTo(channels[2], CV_32FC1, 1.0);

			// x represents row index in the original image;
			int x = 0;
			// y represents collumn index in the original image;;
			int y = 0;
			
			Mat dctMat;
			Mat blocB;
			Mat blocG;
			Mat blocR;
			Scalar intensity1;
			Scalar intensity2;
			Scalar watermarkIntensity;
			// Reading 8x8 bloc from image and inserting the watermark
			while (x + 8 <= channels[2].rows) {
				y = 0;
				while (y + 8 <= channels[2].cols) {
					// Reading 8x8 bloc;
					blocB = Mat(channels[0], Rect(y, x, 8, 8));
					blocG = Mat(channels[1], Rect(y, x, 8, 8));
					blocR = Mat(channels[2], Rect(y, x, 8, 8));
					// Processing DCT of the bloc;
					dct(blocB, blocB);
					dct(blocG, blocG);
					dct(blocR, blocR);
					// Reading the watermark intensity at position j;
					watermarkIntensity = watermarkArray.at(j);
					intensity1 = blocB.at<float>(5, 2);
					intensity2 = blocB.at<float>(4, 3);
					//cout << "8x8 : " << blocB.at<float>(5, 5) << endl;
					// Watermark insertion;
					
					if (watermarkIntensity.val[0] == 0) {

						blocB.at<float>(5, 5) = alpha;
						blocG.at<float>(5, 5) = alpha;
						blocR.at<float>(5, 5) = alpha;

						
						/*if (intensity1.val[0] < intensity2.val[0]) {
					
						
							blocB.at<float>(5, 2) = 20;
							blocB.at<float>(4, 3) = -20;
							blocG.at<float>(5, 2) = 20;
							blocG.at<float>(4, 3) = -20;
							blocR.at<float>(5, 2) = 20;
							blocR.at<float>(4, 3) = -20;
						}*/
					}
					else {
						
						blocB.at<float>(5, 5) = -alpha;
						blocG.at<float>(5, 5) = -alpha;
						blocR.at<float>(5, 5) = -alpha;
						/*
						if (intensity1.val[0] >= intensity2.val[0]) {
							
							blocB.at<float>(5, 2) = -20;
							blocB.at<float>(4, 3) = 20;
							blocG.at<float>(5, 2) = -20;
							blocG.at<float>(4, 3) = 20;
							blocR.at<float>(5, 2) = -20;
							blocR.at<float>(4, 3) = 20;
						}*/
					}
					//cout << "8x8 : " << blocB.at<float>(5, 5) << endl;
					//cout << bloc.at<float>(5, 2) << endl;
					//cout << bloc.at<float>(4, 3) << endl;
					// Transforming the bloc back into spatial domain;
					dct(blocB, blocB, DCT_INVERSE);
					dct(blocG, blocG, DCT_INVERSE);
					dct(blocR, blocR, DCT_INVERSE);
					// Go to the next watermark pixel;
					j = j + 1;
					// if it is the last pixel then circle back to the begining;
					if (j == watermarkArray.size()) {
						j = 0;
						redundancy = redundancy + 1;
						cout << "Redundancy : " << redundancy << '\n';
					}
					// Next Bloc
					y = y + 8;
				}
				// Next bloc
				x = x + 8;
			}

			// Reconstructing the image and saving it
			channels[0].convertTo(channels[0], CV_8UC1, 1.0);
			channels[1].convertTo(channels[1], CV_8UC1, 1.0);
			channels[2].convertTo(channels[2], CV_8UC1, 1.0);
			merge(channels, 3, image);
			
			
			originalImages.at(i) = image;
			cout << "Row : " << x << endl;
			cout << "Col : " << y << endl;
			// Go to the next image
			i = i + 1;
		}
		if (redundancy == 0) {
			cout << "Rendundancy is null: Watermark insertion failed due to : insufficient image dimensions" << std::endl;
			return -1;
		}

		//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		//imshow("Display window", originalImages.at(0)); // Show our image inside it.
		//waitKey(0); // Wait for a keystroke in the window
		// Write extracted watermarks
		for (int i = 0; i < originalImages.size(); ++i) {
			std::string name = std::string("watermarked" + std::to_string(i));
			std::string ext = std::string(".jpg");
			imwrite(name + ext, originalImages.at(i));
		}

		return 0;
		/*
		cout << "Number of channels : " << image.channels() << std::endl;
		Mat channels[3];
		split(image, channels);
		// Converting blue channel of image to float
		channels[2].convertTo(channels[2], CV_32F, 1.0 / 255);


		int i = 0;
		int j = 0;
		int k = 0;
		int cpt = 0;
		Mat dctMat;
		Mat bloc;
		Scalar intensity1;
		Scalar intensity2;
		Scalar watermarkIntensity;

		while (i + 8 <= channels[2].rows && k < watermarkArray.size()) {
			j = 0;
			while (j + 8 <= channels[2].cols && k < watermarkArray.size()) {
				bloc = Mat(channels[2], Rect(j, i, 8, 8));
				dct(bloc, bloc);

				watermarkIntensity = watermarkArray.at(k);
				intensity1 = bloc.at<float>(5, 2);
				intensity2 = bloc.at<float>(4, 3);
				if (watermarkIntensity.val[0] == 0) {
					if (intensity1.val[0] < intensity2.val[0]) {

						float temp;
						temp = bloc.at<float>(5, 2);
						bloc.at<float>(5, 2) = bloc.at<float>(4, 3);
						bloc.at<float>(4, 3) = temp;
					}
				}
				else {
					if (intensity1.val[0] > intensity2.val[0]) {

						float temp;
						temp = bloc.at<float>(5, 2);
						bloc.at<float>(5, 2) = bloc.at<float>(4, 3);
						bloc.at<float>(4, 3) = temp;
					}
				}

				dct(bloc, bloc, DCT_INVERSE);

				k = k + 1;
				j = j + 8;
			}

			i = i + 8;
		}
		channels[2].convertTo(channels[2], CV_8U, 255 / 1.0);
		cout << "number of pixels inserted : " << k << std::endl;


		namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("Display window", channels[2]); // Show our image inside it.
		*/
	};

	int extractWatermark(std::vector<std::string> imagesPath, int width, int heigth) {
		
		// extracted watermarks
		std::vector<Mat> watermarks;
		// Variable that stores an extracted watermark;
		Mat watermark = Mat(width, heigth, CV_8U, Scalar(0));
		watermark.at<uchar>(0, 0) = 0;
		int watermarkSize = width * heigth;
		
		/* ----------------   Loading images  ---------------------- */
		int imgNumber = imagesPath.size();
		std::vector<Mat> originalImages;
		int i = 0;
		while (i < imgNumber) {
			Mat imageread = imread(imagesPath.at(i), IMREAD_COLOR);
			if (!imageread.data) {
				cout << "Could not open or find the image at : " << imagesPath.at(i) << std::endl;
				return -1;
			}
			originalImages.push_back(imageread);
			i++;
		}
		/* ----------------   Begin watermark extraction ---------------------- */
		imgNumber = originalImages.size();
		int redundancy = 0;
		i = 0;
		// j represents row index of extracted watermark image;
		int j = 0;
		// k represents collumn index of extracted watermark image;
		int k = 0;
		
		while ( i < imgNumber )
		{
			Mat image = originalImages.at(i);
			// Spliting image into Three channels assuming it is an RGB image.
			Mat channels[3];
			split(image, channels);
			
			// Converting the blue channel of the image into float
			channels[0].convertTo(channels[0], CV_32FC1, 1.0);

			// x represents row index in the original image;
			int x = 0;
			// y represents collumn index in the original image;;
			int y = 0;
			
			Mat bloc;
			Scalar intensity1;
			Scalar intensity2;
			
			// Reading 8x8 bloc from image and extracting the watermark
			while (x + 8 <= channels[0].rows) {
				y = 0;
				while (y + 8 <= channels[0].cols) {
					// Reading 8x8 bloc;
					bloc = Mat(channels[0], Rect(y, x, 8, 8));
					// Processing DCT of the bloc;
					dct(bloc, bloc);
					
					// Reading the watermark intensity at position j;
					intensity1 = bloc.at<float>(5, 5);
					intensity2 = bloc.at<float>(4, 3);
					//cout << bloc.at<float>(5, 2) << endl;
					//cout << bloc.at<float>(4, 3) << endl;
					// Watermark extraction;
					
					cout << intensity1.val[0] << endl;
					if (intensity1.val[0] > 0) {
						watermark.at<uchar>(k, j) = 0;
					}
					else {
						watermark.at<uchar>(k, j) = 255;
					}
					

					// Go to the next watermark pixel;
					j = j + 1;
					if (j == watermark.cols) {
						j = 0;
						k = k + 1;
						if (k == watermark.rows) {
							k = 0;
							redundancy = redundancy + 1;
							Mat extractedWatermark = Mat(width, heigth, CV_8U, 0);
							watermark.copyTo(extractedWatermark);
							watermarks.push_back(extractedWatermark);
							cout << "Redundancy : " << redundancy << '\n';
						}
					}
					
					// Next Bloc
					y = y + 8;
				}
				// Next bloc
				x = x + 8;
			}
			// Next Image
			i = i + 1;
		}
		// Write extracted watermarks
		for (int i = 0; i < watermarks.size(); ++i) {
			std::string name = std::string("watermark" + std::to_string(i));
			std::string ext = std::string(".jpg");
			imwrite(name + ext, watermarks.at(i));
		}
		/*namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("Display window", watermarks.at(0)); // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window*/
		

		return 0;
	}
}

int main(int argc, char** argv)
{
	if (argc <= 1)
	{
		cout << "Usage: [Image] [Image]* ( -watermark [Image] || -extract )" << endl;
		return -1;
	}

	int i = 1;
	std::string param = std::string(argv[i]);
	std::vector<std::string> imagesPath;
	while (i < argc) {
		param = std::string(argv[i]);
		if (param == "-watermark" || param == "-extract") {
			break;
		}
		imagesPath.push_back(param);
		i = i + 1;
	}

	if (imagesPath.size() == 0) {
		cout << " Specify Images to watermark" << endl;
		return -1;
	} 

	if (i >= argc) {
		cout << "Specify weither you want to insert or extract a watermark : use the options -watermark or -extract" << endl;
		return -1;
	}

	if (param == "-watermark") {
		if (i + 1 >= argc) {
			cout << "Specify the watermark image path";
			return -1;
		}
		i = i + 1;
		std::string watermarkPath = std::string(argv[i]);
		float alpha = 50;

		i = i + 1;
		if (i + 1 < argc) {
			param = std::string(argv[i]);
			if (param == "-alpha") {
				i = i + 1;
				std::string alphaStr = std::string(argv[i]);
				try {
					alpha = std::stoi(alphaStr);
				}
				catch (const std::exception& e) {
					cout << "alpha must be an Integer" << std::endl;
					return -1;
				}
			}
		}

		return watermark::embedWatermark(imagesPath, watermarkPath, alpha);
	}
	else {
		if (param == "-extract") {
			i = i + 1;
			if (i >= argc) {
				cout << "Specify watermark image width" << std::endl;
				return -1;
			}
			std::string widthString = std::string(argv[i]);
			i = i + 1;
			if (i >= argc) {
				cout << "Specify watermark image heigth" << std::endl;
				return -1;
			}
			std::string heigthString = std::string(argv[i]);
			int width;
			int heigth;
			try {
				width = std::stoi(widthString);
				heigth = std::stoi(heigthString);
			}
			catch (const std::exception& e) {
				cout << "Width and Heigth must be Integers" << std::endl;
				return -1;
			}
			
			return watermark::extractWatermark(imagesPath, width, heigth);
		}
	}
};




/*Scalar intensity;
for (int d = 0; d < watermarkArray.size(); ++d) {
watermarkArray.at(d) = 0;
intensity = watermarkArray.at(d);
cout << "element : " << intensity.val[0] << '\n';
};*/

/*
Mat small = Mat(image, Rect(0, 0, 512, 512)).clone();
Mat imageBlocks[2];
imageBlocks[0] = small;*/

// Converting blue channel of image to float
/*Mat blueChannel;
channels[2].convertTo(blueChannel, CV_32F, 1.0 / 255);
Mat dctMat;

cout << array.size() << std::endl;*/

//dct(fimage, dimage);
// Do processing
/*dct(dimage, dimage, DCT_INVERSE);
dimage.convertTo(dimage, CV_8U, 255 / 1.0);*/

/*Mat resultImage;

merge(channels, 3, resultImage);*/