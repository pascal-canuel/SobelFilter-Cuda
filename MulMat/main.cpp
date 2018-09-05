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

int Gx[3][3] = {{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}};

int Gy[3][3] = {{-1, -2, -1},
				{0, 0, 0},
				{1, 2, 1}};

int gradientX(Mat pImg, Point pPos) {
	int total = 0;
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			int i = pPos.y + y - 1;
			int j = pPos.x + x - 1;
			int current = (int)pImg.at<uchar>(i, j);
			int g = current * Gx[y][x];
			total += g;
		}
	}

	return total;
}

int gradientY(Mat pImg, Point pPos) {
	int total = 0;
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			int i = pPos.y + y - 1;
			int j = pPos.x + x - 1;
			int current = (int)pImg.at<uchar>(i, j);
			int g = current * Gx[y][x];
			total += g;
		}
	}
	
	return total;
}

//int convolutionMat(int *pFiltre, Point pPos) {
//	
//}

int approxGradient(int pGradientX, int pGradientY) {
	return abs(pGradientX) + abs(pGradientY);
}

int main()
{
	String imgPath = "../picture/lenaB.jpg";
	Mat imgInput = imread(imgPath, 0);
	Mat imgOutput = imread(imgPath, 0);

	imshow("lenaInput", imgInput);

	//	Call kernel launcher
	int k = 50;
	Launcher_ScalaireMulMat_Int(imgInput.data, k, imgOutput.data, dim3(imgInput.rows, imgOutput.cols));

	//	imgOutput.data should point to the modified data
	imshow("lenaOutput", imgOutput);

	//	Sobel filter on CPU
	Mat imgSobelCPU = imread(imgPath, 0);
	for (int y = 1; y < imgSobelCPU.rows - 1; y++) {
			for (int x = 1; x < imgSobelCPU.cols - 1; x++) {
				int current = (int)imgSobelCPU.at<uchar>(y, x);
				Point pos = Point(x, y);
				int gx = gradientX(imgSobelCPU, pos);
				int gy = gradientY(imgSobelCPU, pos);
				int approx = approxGradient(gx, gy);
				if(approx > 255)
					imgSobelCPU.at<uchar>(y, x) = 255;
				else
					imgSobelCPU.at<uchar>(y, x) = approx;
				
			}
	}

	imshow("SobelCPU", imgSobelCPU);

	waitKey(0);
	return 0;
}


