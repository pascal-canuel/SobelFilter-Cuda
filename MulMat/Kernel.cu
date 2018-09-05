#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <stdio.h>

typedef unsigned char uchar;

//	If you want to have the .cu intellisense:
//	Adding '.cu' under c++ extension in visual studio settings would enable syntax highlighting for c++ 
//	keywords only. EDIT: It in Tools -> Options -> Text Editor -> File Extension type in cu and select 
//	Microsoft Visual C++ as the editor and click add

/************************************************************************
// KERNEL qui permet de faire une multiplication scalaire d'une matrice
// d'entier. Chaque thread s'occupe d'un résultat
/***********************************************************************/
__global__
static void Kernel_ScalaireMulMat_Int(uchar *MatI, int K, uchar *MatO)
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	int ImageWidth = blockDim.x * gridDim.x;
	int Index = ImgNumLigne * ImageWidth + ImgNumColonne;

	MatO[Index] = MatI[Index] + 50;
}

/************************************************************************
// Fonction de lancement du kernel qui permet de faire une multiplication
// scalaire d'une matrice d'entier.
/***********************************************************************/
extern "C" cudaError_t Launcher_ScalaireMulMat_Int(uchar *pMatI, int K, uchar *pMatO, dim3 DimMat)
{
	//	1. Initialize data
	//	Choose which GPU to run on, change this on a multi-GPU system.    
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	int BLOCK_SIZE = 16;
	uchar *gMatI, *gMatO;

	//	Grid of BLOCK_SIZE * BLOCK_SIZE blocks
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	//	Block of BLOCK_SIZE * BLOCK_SIZE threads
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	size_t memSize = DimMat.x * DimMat.y * sizeof(uchar);

	//	2. Allocate memory for the data on the GPU
	cudaStatus = cudaMalloc(&gMatI, memSize);
	cudaStatus = cudaMalloc(&gMatO, memSize);

	//	3. Copy the data on the GPU
	cudaStatus = cudaMemcpy(gMatI, pMatI, memSize, cudaMemcpyHostToDevice);

	//	4. Launch kernel
	Kernel_ScalaireMulMat_Int <<<dimGrid, dimBlock >>>(gMatI, K, gMatO);
	cudaStatus = cudaDeviceSynchronize();	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel failed!");
		return cudaStatus;
	}

	//	5. Copy the data back on the CPU
	cudaMemcpy(pMatO, gMatO, memSize, cudaMemcpyDeviceToHost);

	//	6. Free the memory of the GPU
	cudaFree(gMatI);
	cudaFree(gMatO);

	return cudaStatus;
}
