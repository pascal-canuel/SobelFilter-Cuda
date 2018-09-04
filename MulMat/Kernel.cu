#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <stdio.h>

typedef unsigned char uchar

//	If you want to have the .cu intellisense:
//	Adding '.cu' under c++ extension in visual studio settings would enable syntax highlighting for c++ 
//	keywords only. EDIT: It in Tools -> Options -> Text Editor -> File Extension type in cu and select 
//	Microsoft Visual C++ as the editor and click add

extern "C" cudaError_t ConvolutionCuda(int *ImageIn,  int *ImageOut, int *Kernel, size_t ImageSize) 
{ 
   // Choose which GPU to run on, change this on a multi-GPU system.    
   cudaError_t cudaStatus = cudaSetDevice(0);    


   if (cudaStatus != cudaSuccess) {   
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  
		goto Error;  
   } 
   
   Error:
	return cudaStatus;
}

/************************************************************************
// KERNEL qui permet de faire une multiplication scalaire d'une matrice
// d'entier. Chaque thread s'occupe d'un résultat
/***********************************************************************/
__global__ 
static void Kernel_ScalaireMulMat_Int(int *MatA, int K, int *MatR, dim3 DimMat)
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	int ImageWidth = blockDim.x * gridDim.x;
	int Index = ImgNumLigne * ImageWidth + ImgNumColonne;

	MatR[Index] = MatA[Index] + K;
	return;
}

/************************************************************************
// Fonction de lancement du kernel qui permet de faire une multiplication
// scalaire d'une matrice d'entier.
/***********************************************************************/
extern "C" cudaError_t Launcher_ScalaireMulMat_Int(int *pMatA, int K, int *pMatR, dim3 DimMat)
{
	int BLOCK_SIZE = 32; //	Should be defined
	int *MatA, *MatR;
	dim3 dimBlock(DimMat.x, DimMat.y);
	//dim3 dimGrid(iDivUp(DimMat.x, BLOCK_SIZE), iDivUp(DimMat.y, BLOCK_SIZE)); 	
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	cudaError_t cudaStatus; 
	// Partir un timer pour calculer le temps d'exécution 
	unsigned int timer = 0; float TempsExecution;  
	// Allouer l'espace memoire des 2 matrices sur la carte GPU 
	size_t memSize = DimMat.x * DimMat.y * sizeof(int); 
	cudaMalloc( (void **) &MatA, memSize ); 
	cudaMalloc( (void **) &MatR, memSize ); 
	// Copier de la matrice A dans la memoire du GPU 
	cudaMemcpy( MatA, pMatA, memSize, cudaMemcpyHostToDevice ); 
	// Partir le kernel. ************* Sur une seul ligne  ********* 
	Kernel_ScalaireMulMat_Int<<<dimGrid,dimBlock>>>((int*)MatA,(int)K,(int*)MatR, DimMat);
	//CUT_CHECK_ERROR("Kernel execution failed\n"); 
	// Attendre la fin du kernel  
	cudaStatus = cudaDeviceSynchronize();  
	if (cudaStatus != cudaSuccess) {   
		fprintf(stderr, "Kernel  BackGroundSoustractionHSV failed!");
		return cudaStatus;  
	}
	// Transfert de la matrice résultat 
	//CUDA_SAFE_CALL( cudaMemcpy(pMatR, MatR, memSize, cudaMemcpyDeviceToHost));
	cudaMemcpy(pMatR, MatR, memSize, cudaMemcpyDeviceToHost);
	// Libérer la mémoire du 
	//GPU CUDA_SAFE_CALL( cudaFree(MatA)); 
	cudaFree(MatA);
	return cudaStatus;
}