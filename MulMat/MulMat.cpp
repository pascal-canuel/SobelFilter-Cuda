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

extern "C" cudaError_t ConvolutionCuda(int *ImageIn, int *ImageOut, int *Kernel, size_t ImageSize);
extern "C" cudaError_t Launcher_ScalaireMulMat_Int(int *pMatA, int K, int *pMatR, dim3 DimMat);

extern  float TempsExecution;

int main()
{
	Mat img = cv::imread("../picture/lena.jpg", 0);
	int pixTotalIteration = 0;
	imshow("lenaIn", img);
	
	/*for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			img.at<uchar>(y, x) += 50;
			std::cout << (int)img.at<uchar>(y, x) << std::endl;
			pixTotalIteration += 1;
		}
	}

	std::cout << std::endl << "Iterate through: " << pixTotalIteration << " pixels";
	std::cout << std::endl << " Rows: " << img.rows << " Cols: " << img.cols << " Should iterate through: " << img.cols * img.rows << std::endl;

	imshow("lenaOut", img);*/

	extern Mat MatOut;
	dim3 imageSize = dim3(img.cols, img.rows);
	int k = 10;
	
	cudaError_t cudaStatus = Launcher_ScalaireMulMat_Int((int*)&img.data, k, (int*)&MatOut.data, imageSize);

	waitKey(0);
    return 0;
}

