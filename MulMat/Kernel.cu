#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <stdio.h>

//	If you want to have the .cu intellisense:
//	Adding '.cu' under c++ extension in visual studio settings would enable syntax highlighting for c++ 
//	keywords only. EDIT: It in Tools -> Options -> Text Editor -> File Extension type in cu and select 
//	Microsoft Visual C++ as the editor and click add

extern "C" cudaError_t ConvolutionCuda(int *ImageIn,  int *ImageOut, int *Kernel, size_t ImageSize) 
{ 
	int *MatIn, *MatOut;
   // Choose which GPU to run on, change this on a multi-GPU system.    
   cudaError_t cudaStatus = cudaSetDevice(0);    
   if (cudaStatus != cudaSuccess) {   
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  
		goto Error;  
   } 
   
   Error:
	return cudaStatus;
}

