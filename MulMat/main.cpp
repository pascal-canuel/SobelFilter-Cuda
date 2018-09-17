// VisionLab1.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
// Lab 1 Vision par ordinateur
// Par Pascal Canuel et Justin Roberge-Lavoie

//#include "pch.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//#include "range.h"

// MulMat.cpp : définit le point d'entrée pour l'application console.
//
#include "stdafx.h"

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cmath>
#include <chrono>  // for high_resolution_clock

using namespace cv;

extern "C" bool GPGPU_Sobel(cv::Mat* imgTresh, cv::Mat* Grayscale);
extern  float TempsExecution; //	TODO start timer

Mat imgSobelCPU;
Mat imgSobelCPUNorm;
Mat imgInput;

int Gx[3][3] = { {-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1} };

int Gy[3][3] = { {-1, -2, -1},
				{0, 0, 0},
				{1, 2, 1} };

int GxF[3][3] = { {-3, 0, 3},
				{-10, 0, 10},
				{-3, 0, 3} };

int GyF[3][3] = { {-3, -10, -3},
				{0, 0, 0},
				{3, 10, 3} };

int gradient(int posY, int posX) {
	int totalX = 0;
	int totalY = 0;
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			int current = (int)imgSobelCPU.at<uchar>(posY + y, posX + x);
			int conX = current * Gx[y][x];
			int conY = current * Gy[y][x];
			totalX += conX;
			totalY += conY;
		}
	}
	return abs(totalX) + abs(totalY);
}

int gradientNorm(int posY, int posX) {
	int totalX = 0;
	int totalY = 0;
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			int current = (int)imgInput.at<uchar>(posY + y, posX + x);
			int conX = current * GxF[y][x];
			int conY = current * GyF[y][x];
			totalX += conX;
			totalY += conY;
		}
	}
	return abs(totalX) + abs(totalY);
}

int main()
{
	String imgPath = "../picture/city.jpg";
	imgInput = imread(imgPath, 0);
	//Mat imgOutput = imread(imgPath, 0);

	imshow("InputGray", imgInput);

	Mat img = imread(imgPath);
	//namedWindow("Original", WINDOW_NORMAL);
	//resizeWindow("Original", 1800, 900);
	imshow("Original", img);
	//	Call kernel launcher
	//int k = 50;
	//Launcher_ScalaireMulMat_Int(imgInput.data, k, imgOutput.data, dim3(imgInput.rows, imgOutput.cols));

	//	imgOutput.data should point to the modified data
	//imshow("lenaOutput", imgOutput);

	//	Sobel filter on CPU
	//imgSobelCPU = imread(imgPath, 0);

	//int size = (imgSobelCPU.rows - 2) * (imgSobelCPU.cols - 2);
	//int *gradTotal = new int[size];
	//for (int y = 0; y < imgSobelCPU.rows - 2; y++) {
	//	for (int x = 0; x < imgSobelCPU.cols - 2; x++) {
	//		int width = imgSobelCPU.cols - 2;
	//		int i = width * y + x;

	//		int g = gradient(y, x);
	//		gradTotal[i] = g;
	//	}
	//}

	//int minVal, maxVal;
	//minVal = *std::min_element(gradTotal, gradTotal + size);
	//maxVal = *std::max_element(gradTotal, gradTotal + size);

	//// mapper int entre 0 et 255
	//for (int y = 0; y < imgSobelCPU.rows - 2; y++) {
	//	for (int x = 0; x < imgSobelCPU.cols - 2; x++) {

	//		int width = imgSobelCPU.cols - 2;
	//		int i = width * y + x;
	//		int current = gradTotal[i];

	//		int map = (current * 255) / maxVal;

	//		imgSobelCPU.at<uchar>(y, x) = map;
	//	}
	//}

	//imshow("SobelCPU", imgSobelCPU);
	Mat gpuSobel = imread(imgPath, 0);
	GPGPU_Sobel(&imgInput, &gpuSobel);

	//namedWindow("SobelGPU", WINDOW_NORMAL);
	//resizeWindow("SobelGPU", 1800, 900);
	imshow("SobelGPU", gpuSobel);

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	
	imgSobelCPUNorm = imread(imgPath, 0);
	for (int y = 0; y < imgSobelCPUNorm.rows - 2; y++) {
		for (int x = 0; x < imgSobelCPUNorm.cols - 2; x++) {

			int i = imgSobelCPUNorm.step1()  * y + x;

			/*if (y == 0 || x == 0) {
				imgSobelCPUNorm.data[i] = 255;
			}*/

			/*if ( y == imgSobelCPUNorm.rows - 1 || x == imgSobelCPUNorm.cols - 1) {
				imgSobelCPUNorm.data[i] = 255;
			}
			else if (y == imgSobelCPUNorm.rows - 2 || x == imgSobelCPUNorm.cols - 2) {

			}
			else {*/
				int width = imgSobelCPUNorm.cols - 2;

				int g = gradientNorm(y, x);

				int norm = g * 0.0625;

				//imgSobelCPUNorm.at<uchar>(y + 1, x + 1) = norm;
				imgSobelCPUNorm.data[i] = norm;
			//}
						
		}
	}
	imshow("norm", imgSobelCPUNorm);
	
	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time CPU: " << elapsed.count() << " s\n";
	
	//	Expected Sobel
	Mat src_gray = imread(imgPath, 0);
	Mat grad;
	int scale = 1;
	int delta = 0;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ddepth = CV_16S;

	/// Gradient X
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	/// Gradient Y
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow("OpenCVSobel", grad);

	waitKey(0);
	return 0;
}


