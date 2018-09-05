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

using namespace cv;

extern "C" cudaError_t Launcher_ScalaireMulMat_Int(uchar *pMatI, int K, uchar *pMatO, dim3 DimMat);

extern  float TempsExecution;

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

	waitKey(0);
	return 0;
}

