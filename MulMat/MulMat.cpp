// MulMat.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 

using namespace cv;

extern "C" cudaError_t ConvolutionCuda(int *ImageIn, int *ImageOut, int *Kernel, size_t ImageSize);

extern  float TempsExecution;

int main()
{
	Mat img = cv::imread("../picture/lena.jpg", 0);
	imshow("lena", img);
	waitKey(0);
    return 0;
}

