#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cuda_runtime.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "stdafx.h"
//#include "nppdefs.h"
//#include <npp.h>
#include <chrono>  // for high_resolution_clock

typedef unsigned char uchar;
typedef unsigned int uint;

#define BLOCK_SIZE 32
#define CV_64FC1 double
#define CV_32F float
#define CV_8U uchar

int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__
int absGrad(int grad) {
	if (grad < 0) {
		return -1 * grad;
	}
	else {
		return grad;
	}
}

__global__ void Kernel_Sobel(uchar* img, uchar* imgout, int ImgWidth, int imgHeigh) // , int* maskX, int* maskY
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	
	int Index = (ImgNumLigne * ImgWidth) + (ImgNumColonne); // will be 3x greater

	//int nani = (ImgNumLigne * (ImgWidth / 3)) + ImgNumColonne;

	if ((ImgNumColonne < ImgWidth -2 ) && (ImgNumLigne < imgHeigh -2 )) //width / 3
	{
		//imgout[Index] = 50;

			//int y = ImgNumLigne; // change imgnumligne pour y
			//int x = ImgNumColonne;
			//int i = Index;
			////imgout ->>> int 
			int i = Index;
			int gradX = img[i] * -3 + img[i + 1] * 0 + img[i + 2] * 3;
			i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne);
			gradX += img[i] * -10 + img[i + 1] * 0 + img[i + 2] * 10;
			i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne);
			gradX += img[i] * -3 + img[i + 1] * 0 + img[i + 2] * 3;
			
			i = Index;
			int gradY = img[i] * -3 + img[i + 1] * -10 + img[i + 2] * -3;
			i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne);
			gradY += img[i] * 0 + img[i + 1] * 0 + img[i + 2] * 0;
			i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne);
			gradY += img[i] * 3 + img[i + 1] * 10 + img[i + 2] * 3;


			int grad = absGrad(gradX) + absGrad(gradY);
			int norm = grad * 0.0625;

			imgout[Index] = norm;
			////	Gradient X ne pas calculer * 0
			//int gradX = img[i] * -1 + img[i + 1] * 0 + img[i + 2] * 1;
			//i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne);
			//gradX += img[i] * -2 + img[i + 1] * 0 + img[i + 2] * 2;
			//i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne);
			//gradX += img[i] * -1 + img[i + 1] * 0 + img[i + 2] * 1;

			//i = (ImgNumLigne * ImgWidth) + (ImgNumColonne * 3);

			////	Gradient Y
			//int gradY = img[i] * -1 + img[i + 1] * -2 + img[i + 2] * -1;
			//i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne);
			//gradY += img[i] * 0 + img[i + 1] * 0 + img[i + 2] * 0;
			//i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne);
			//gradY += img[i] * 1 + img[i + 1] * 2 + img[i + 2] * 1;

			////	Gradient 
			//int gradient = abs(gradX) + abs(gradY);
			//int norm = gradient * 0.125;

			//imgout[i] = norm;
	}

	return;
}

extern "C" bool GPGPU_Sobel(cv::Mat* imgTresh, cv::Mat* Grayscale)
{

	//	1. Initialize data
	cudaError_t cudaStatus;
	uchar* gDevImage;
	uchar* gDevImageOut;

	uint imageSize = imgTresh->rows * imgTresh->step1(); // will be x 3 greater 
	uint gradientSize = imgTresh->rows * imgTresh->cols * sizeof(uchar);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(iDivUp(imgTresh->cols, BLOCK_SIZE), iDivUp(imgTresh->rows, BLOCK_SIZE));

	//	2. Allocation data
	cudaStatus = cudaMalloc(&gDevImage, imageSize);
	cudaStatus = cudaMalloc(&gDevImageOut, gradientSize);

	//	3. Copy data on GPU
	cudaStatus = cudaMemcpy(gDevImage, imgTresh->data, imageSize, cudaMemcpyHostToDevice);

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	
	//	4. Launch kernel
	Kernel_Sobel<<<dimGrid, dimBlock>>>(gDevImage, gDevImageOut, imgTresh->step1(), imgTresh->rows);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//Wait for the kernel to end
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		goto Error;
	}

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time GPGPU: " << elapsed.count() << " s\n";
	
	//	5. Copy data on CPU
	cudaStatus = cudaMemcpy(Grayscale->data, gDevImageOut, gradientSize, cudaMemcpyDeviceToHost);

	//	6. Free GPU memory
Error:
	cudaFree(gDevImage);
	cudaFree(gDevImageOut);

	return cudaStatus;
}
