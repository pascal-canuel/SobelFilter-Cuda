// MulMat.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cmath>

using namespace cv;

extern "C" cudaError_t Launcher_ScalaireMulMat_Int(uchar *pMatI, int K, uchar *pMatO, dim3 DimMat);
extern  float TempsExecution; //	TODO start timer

Mat imgSobelCPU;

int Gx[3][3] = {{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}};

int Gy[3][3] = {{-1, -2, -1},
				{0, 0, 0},
				{1, 2, 1}};

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

int main()
{
	String imgPath = "../picture/lenaB.jpg";
	Mat imgInput = imread(imgPath, 0);
	Mat imgOutput = imread(imgPath, 0);

	imshow("lenaInput", imgInput);

	//	Call kernel launcher
	int k = 50;
	//Launcher_ScalaireMulMat_Int(imgInput.data, k, imgOutput.data, dim3(imgInput.rows, imgOutput.cols));

	//	imgOutput.data should point to the modified data
	//imshow("lenaOutput", imgOutput);

	//	Sobel filter on CPU
	imgSobelCPU = imread(imgPath, 0);
	
	int size = (imgSobelCPU.rows - 2) * (imgSobelCPU.cols - 2);
	int *gradTotal = new int[size];
	for (int y = 0; y < imgSobelCPU.rows - 2; y++) {
		for (int x = 0; x < imgSobelCPU.cols - 2; x++) {
			int width = imgSobelCPU.cols - 2;
			int i = width * y + x;

			int g = gradient(y, x);
			gradTotal[i] = g;
		}
	}

	int minVal, maxVal;
	minVal = *std::min_element(gradTotal, gradTotal + size);
	maxVal = *std::max_element(gradTotal, gradTotal + size);
	
	// mapper int entre 0 et 255
	for (int y = 0; y < imgSobelCPU.rows - 2; y++) {
		for (int x = 0; x < imgSobelCPU.cols - 2; x++) {

			int width = imgSobelCPU.cols - 2;
			int i = width * y + x;
			int current = gradTotal[i];

			int map = (current * 255) / maxVal;

			imgSobelCPU.at<uchar>(y, x) = map;
		}
	}

	imshow("SobelCPU", imgSobelCPU);

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
	imshow("ExpectedSobel", grad);

	waitKey(0);
	return 0;
}


