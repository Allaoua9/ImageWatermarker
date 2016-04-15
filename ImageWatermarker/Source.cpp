#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

namespace watermark {

	int embedWatermark(std::vector<std::string> imagesPath, string watermarkPath) {

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
		// loop through all the images to insert the watermark
		while ( i < imgNumber) {
			Mat image = originalImages.at(i);
			// Spliting image into Three channels assuming it is an RGB image.
			Mat channels[3];
			split(image, channels);
			// Converting the blue channel of the image into float
			channels[2].convertTo(channels[2], CV_32F, 1.0 / 255);

			// x represents row index in the original image;
			int x = 0;
			// y represents collumn index in the original image;;
			int y = 0;
			// j represents index of current watermark pixel;
			int j = 0;
			Mat dctMat;
			Mat bloc;
			Scalar intensity1;
			Scalar intensity2;
			Scalar watermarkIntensity;
			// Reading 8x8 bloc from image and inserting the watermark
			while (x + 8 <= channels[2].rows) {
				y = 0;
				while (y + 8 <= channels[2].cols) {
					// Reading 8x8 bloc;
					bloc = Mat(channels[2], Rect(y, x, 8, 8));
					// Processing DCT of the bloc;
					dct(bloc, bloc);
					// Reading the watermark intensity at position j;
					watermarkIntensity = watermarkArray.at(j);
					intensity1 = bloc.at<float>(5, 2);
					intensity2 = bloc.at<float>(4, 3);
					// Watermark insertion;
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
					// Transforming the bloc back into spatial domain;
					dct(bloc, bloc, DCT_INVERSE);
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

			// Reconstruction the image and saving it
			channels[2].convertTo(channels[2], CV_8U, 255 / 1.0);
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

		Mat watermark = Mat(width, heigth, CV_8U, 0);



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
		return watermark::embedWatermark(imagesPath, watermarkPath);
	}
	else {
		if (param == "-extract") {
			return watermark::extractWatermark(imagesPath);
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