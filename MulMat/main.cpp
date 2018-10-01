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

int main()
{
	String imgPath = "../picture/city.jpg";

	Mat imgInput = imread(imgPath, 0); // load image as grayscale
	Mat gpuSobel = imgInput.clone(); 
	GPGPU_Sobel(&imgInput, &gpuSobel); 
	imshow("SobelGPU", gpuSobel);

	waitKey(0);
	return 0;
}


