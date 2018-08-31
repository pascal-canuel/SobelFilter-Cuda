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
	int pixTotalIteration = 0;
	imshow("lena", img);
	
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			std::cout << (int)img.at<uchar>(y, x) << std::endl;
			pixTotalIteration += 1;
		}
	}

	std::cout << std::endl << "Iterate through: " << pixTotalIteration << " pixels";
	std::cout << std::endl << " Rows: " << img.rows << " Cols: " << img.cols << " Should iterate through: " << img.cols * img.rows << std::endl;
	waitKey(0);
    return 0;
}

