#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "json\json.h"

using namespace cv;
using namespace std;

namespace watermark {

	int embedWatermark(std::vector<std::string> imagesPath, string watermarkPath, float alpha) {

		Json::Value result;

		/* ----------------   Loading images  ---------------------- */
		int imgNumber = imagesPath.size();
		std::vector<Mat> originalImages;
		int i = 0;
		while (i < imgNumber) {
			Mat imageread = imread(imagesPath.at(i), IMREAD_COLOR);
			if (!imageread.data) {
				result["error"]["message"] = "Could not open or find the image at : " + imagesPath.at(i);
				std::cerr << result << std::endl;
				return -1;
			}
			originalImages.push_back(imageread);
			i++;
		}

		Mat watermark = imread(watermarkPath, IMREAD_GRAYSCALE);
		if (!watermark.data) {
			result["error"]["message"] = "Could not open or find the watermark image at : " + watermarkPath;
			std::cerr << result << std::endl;
			return -1;
		}
		/* ------------------------------- Begining watermark embedding----------------------------- */

		// Preprocessing watermark
		// Applying Treshold
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
		// the number of images that have been watermarked;
		int watermarkedNumber = 0;
		// Used to calculate the redundancy of the inserted watermark
		int redundancy = 0;
		// Represents the index of the image to be watermarked
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
			// RGB 8x8 Blocs of the image
			Mat blocB;
			Mat blocG;
			Mat blocR;
			// Used to store the intensity of the DCT Matrix
			Scalar intensity;
			// Used to store the intensity of the watermark pixel.
			Scalar watermarkIntensity;
			// Reading 8x8 bloc from image and inserting the watermark
			while (x + 8 <= channels[2].rows) {
				y = 0;
				while (y + 8 <= channels[2].cols) {
					// Reading 8x8 blocs;
					blocB = Mat(channels[0], Rect(y, x, 8, 8));
					blocG = Mat(channels[1], Rect(y, x, 8, 8));
					blocR = Mat(channels[2], Rect(y, x, 8, 8));
					// Processing DCT of the blocs;
					dct(blocB, blocB);
					dct(blocG, blocG);
					dct(blocR, blocR);
					// Reading the watermark intensity at position j;
					watermarkIntensity = watermarkArray.at(j);

					// Watermark insertion;
					if (watermarkIntensity.val[0] == 0) {
						blocB.at<float>(5, 5) = alpha;
						blocG.at<float>(5, 5) = alpha;
						blocR.at<float>(5, 5) = alpha;
					}
					else {
						blocB.at<float>(5, 5) = -alpha;
						blocG.at<float>(5, 5) = -alpha;
						blocR.at<float>(5, 5) = -alpha;
					}
					// Go to the next watermark pixel;
					j = j + 1;
					// if it is the last pixel then circle back to the begining;
					if (j == watermarkArray.size()) {
						j = 0;
						redundancy = redundancy + 1;
						watermarkedNumber = i + 1;
					}

					watermarkIntensity = watermarkArray.at(j);
					if (watermarkIntensity.val[0] == 0) {

						blocB.at<float>(4, 4) = alpha;
						blocG.at<float>(4, 4) = alpha;
						blocR.at<float>(4, 4) = alpha;
					}
					else {

						blocB.at<float>(4, 4) = -alpha;
						blocG.at<float>(4, 4) = -alpha;
						blocR.at<float>(4, 4) = -alpha;

					}

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
						watermarkedNumber = i + 1;
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
			// Go to the next image
			i = i + 1;
		}
		if (redundancy == 0) {
			result["error"]["message"] = "Watermark insertion failed due to : insufficient images size";
			std::cerr << result << std::endl;
			return -1;
		}

		//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		//imshow("Display window", originalImages.at(0)); // Show our image inside it.
		//waitKey(0); // Wait for a keystroke in the window
		// Write extracted watermarks
		for (int i = 0; i < watermarkedNumber; ++i) {
			// this will erase the input image and save the watermarked one.
			result["watermarkedImages"][i] = imagesPath.at(i);
			imwrite(imagesPath.at(i), originalImages.at(i));
		}

		result["success"] = true;
		result["watermarkedNumber"] = watermarkedNumber;
		result["watermarkRedundancy"] = redundancy;
		result["watermarkWidth"] = watermark.cols;
		result["watermarkHeight"] = watermark.rows;

		std::cout << result << std::endl;

		return 0;
	};

	int extractWatermark(std::vector<std::string> imagesPath, int width, int height) {
		
		Json::Value result;
		// extracted watermarks
		std::vector<Mat> watermarks;
		// Variable that stores an extracted watermark;
		Mat watermark = Mat(width, height, CV_8UC1, Scalar(0));
		watermark.at<uchar>(0, 0) = 0;
		int watermarkSize = width * height;
		
		/* ----------------   Loading images  ---------------------- */
		int imgNumber = imagesPath.size();
		std::vector<Mat> originalImages;
		int i = 0;
		while (i < imgNumber) {
			Mat imageread = imread(imagesPath.at(i), IMREAD_COLOR);
			if (!imageread.data) {
				result["error"]["message"] = "Could not open or find the image at : " + imagesPath.at(i);
				std::cerr << result << std::endl;
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
			// y represents collumn index in the original image;
			int y = 0;
			
			Mat bloc;
			Scalar intensity;
			
			// Reading 8x8 bloc from image and extracting the watermark
			while (x + 8 <= channels[0].rows) {
				y = 0;
				while (y + 8 <= channels[0].cols) {
					// Reading 8x8 bloc;
					bloc = Mat(channels[0], Rect(y, x, 8, 8));
					// Processing DCT of the bloc;
					dct(bloc, bloc);
					
					// Reading the bloc intensity at position (5,5);
					intensity = bloc.at<float>(5, 5);

					// Watermark extraction;
					if (intensity.val[0] > 0) {
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
							Mat extractedWatermark = Mat(width, height, CV_8U, 0);
							watermark.copyTo(extractedWatermark);
							watermarks.push_back(extractedWatermark);
						}
					}

					// Reading the watermark intensity at position (4, 4);
					intensity = bloc.at<float>(4, 4);

					// Watermark extraction;
					if (intensity.val[0] > 0) {
						watermark.at<uchar>(k, j) = 0;
					}
					else {
						watermark.at<uchar>(k, j) = 255;
					}

					j = j + 1;
					if (j == watermark.cols) {
						j = 0;
						k = k + 1;
						if (k == watermark.rows) {
							k = 0;
							redundancy = redundancy + 1;
							Mat extractedWatermark = Mat(width, height, CV_8UC1, 0);
							watermark.copyTo(extractedWatermark);
							watermarks.push_back(extractedWatermark);
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
		if (redundancy == 0) {
			result["error"]["message"] = "Watermark insertion failed due to : insufficient images size";
			std::cerr << result << std::endl;
			return -1;
		}
		// Write extracted watermarks
		for (int i = 0; i < watermarks.size(); ++i) {
			std::string name = std::string("watermark" + std::to_string(i));
			std::string ext = std::string(".png");
			std::string path = std::string(name + ext);
			result["watermarks"][i] = path;
			imwrite(name + ext, watermarks.at(i));
		}
		result["success"] = true;
		/*namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("Display window", watermarks.at(0)); // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window*/
		
		std::cout << result << std::endl;

		return 0;
	}
}

int main(int argc, char** argv)
{
	
	if (argc == 1) {
		// Reading JSON data:
		Json::Value root;
		// JSON for error output
		Json::Value error;

		std::string str;
		// Reading JSON string from stdin;
		std::getline(std::cin, str);
		// Parsing the JSON string;
		Json::Reader reader;
		if (reader.parse(str, root)) {
			if (!root.isMember("method")) {
				error["error"]["message"] = "Please specify the method";
				std::cerr << error << std::endl;
				return -1;
			}

			if (root["method"].asString() != "watermark" && root["method"].asString() != "extract") {
				error["error"]["message"] = "Unknown method : " + root["method"].asString();
				std::cerr << error << std::endl;
				return -1;
			}

			std::vector<std::string> imagesPath;

			if (root.isMember("images") && root["images"].isArray()) {
				for (int i = 0; i < root["images"].size(); ++i) {
					imagesPath.push_back(root["images"][i].asString());
				}
			}
			else {
				error["error"]["message"] = "Specify Images to watermark.";
				std::cerr << error << std::endl;
				return -1;
			}

			if (root["method"].asString() == "watermark") {
				if (!root.isMember("watermark")) {
					error["error"]["message"] = "Specify the watermark image.";
					std::cerr << error << std::endl;
					return -1;
				}
				std::string watermarkPath = root["watermark"].asString();
				if (!root.isMember("alpha")) {
					error["error"]["message"] = "Specify the strength factor (alpha)";
					std::cerr << error << std::endl;
					return -1;
				}
				float alpha = root["alpha"].asFloat();
				
				// Embedding watermark
				return watermark::embedWatermark(imagesPath, watermarkPath, alpha);
			}
			else if (root["method"].asString() == "extract") {
				if (!root.isMember("width")) {
					error["error"]["message"] = "Specify the watermark image width.";
					std::cerr << error << std::endl;
					return -1;
				}
				if (!root.isMember("height")) {
					error["error"]["message"] = "Specify the watermark image height.";
					std::cerr << error << std::endl;
					return -1;
				}
				int width = root["width"].asInt();
				int height = root["height"].asInt();
				
				// Extracting watermark
				return watermark::extractWatermark(imagesPath, width, height);
			}
		}
		else {
			error["error"]["message"] = "Failed to parse JSON";
			std::cerr << error << std::endl;
			return -1;
		}
	}
	// Parsing CMD line params
	else {

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
			std::cerr << " Specify Images to watermark" << std::endl;
			return -1;
		}

		if (i >= argc) {
			std::cerr << "Specify weither you want to insert or extract a watermark : use the options -watermark or -extract" << std::endl;
			return -1;
		}

		if (param == "-watermark") {
			if (i + 1 >= argc) {
				std::cerr << "Specify the watermark image path";
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
						std::cerr << "alpha must be an Integer" << std::endl;
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
					std::cerr << "Specify watermark image width" << std::endl;
					return -1;
				}
				std::string widthString = std::string(argv[i]);
				i = i + 1;
				if (i >= argc) {
					std::cerr << "Specify watermark image height" << std::endl;
					return -1;
				}
				std::string heightString = std::string(argv[i]);
				int width;
				int height;
				try {
					width = std::stoi(widthString);
					height = std::stoi(heightString);
				}
				catch (const std::exception& e) {
					std::cerr << "Width and height must be Integers" << std::endl;
					return -1;
				}

				return watermark::extractWatermark(imagesPath, width, height);
			}
		}

	}

	return -1;
	
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