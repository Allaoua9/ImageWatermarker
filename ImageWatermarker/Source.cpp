#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

namespace watermark {

	int embedWatermark(std::vector<std::string> imagesPath, string watermarkPath) {

		const int imgNumber = imagesPath.size();
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
		// Comment
		cout << originalImages.size() << std::endl;
	};
}

int main(int argc, char** argv)
{
	if (argc <= 1)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	int i = 1;
	std::string param = std::string(argv[i]);
	std::vector<std::string> imagesPath;
	while (i < argc) {
		param = std::string(argv[i]);
		if (param == "-watermark") {
			break;
		}
		imagesPath.push_back(param);
		i = i + 1;
	}

	if (imagesPath.size() == 0) {
		cout << " Specify Images to watermark" << endl;
		return -1;
	}

	if (i >= argc || i + 1 >= argc) {
		cout << " Specify watermark image" << endl;
		return -1;
	}

	i = i + 1;
	std::string watermarkPath = std::string(argv[i]);
	
	watermark::embedWatermark(imagesPath, watermarkPath);

	/*
	Mat image;
	image = imread(argv[1], IMREAD_COLOR); // Read the file
	Mat watermark;
	watermark = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	if (!image.data || !watermark.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// Preprocessing watermark
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
	cout << '\n';
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
	
	
	waitKey(0); // Wait for a keystroke in the window*/
	return 0;
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