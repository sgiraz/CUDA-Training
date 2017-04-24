////////////////////////////////////////////////////////////////////////////
//
//
//  EXAMPLE OF WORK-EFFICIENT PREFIX-SCAN CHAPTER 8
//  efficient_Kogge_Stone_scan
//
////////////////////////////////////////////////////////////////////////////
// ++++++++++++++++++++++++++++++++ WORK IN PROGRESS RIVEDERE BENE LE 3 FASI +++++++++++++++++++++++++++++++++++++++++++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

//Assumption: the number of threads will be equal to section elements
#define SECTION_SIZE 32

cudaError_t efficient_Kogge_Stone_scan(float *X, float *Y, unsigned int size);
void sequential_scan(float *x, float *y, int Max_i);
void print_Array(float *A, int size);
int verify_result(float *Y, float *YS, int size);

__device__
void print_Array_device(float *A, int size) {
	for (int i = 0; i < size; i++) {
		printf("%.2f ", A[i]);
	}
	printf("\n\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Simple bad prefix sum 
//! @param X  input data in global memory
//! @param Y  output data in global memory
//! @param InputSize size of input and output data
////////////////////////////////////////////////////////////////////////////////
__global__ void efficient_Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize)
{
	__shared__ float XY[SECTION_SIZE];

	// collaborative load in a coalesced manner
	for (int j = 0; j < SECTION_SIZE; j += blockDim.x) {
		__syncthreads();
		XY[threadIdx.x + j] = X[threadIdx.x + j];
	}
	
	
	// phase 1: scan inner own subsection
	for (int j = 1; j < SECTION_SIZE/blockDim.x ; j++ ) {
		__syncthreads();
		XY[threadIdx.x * blockDim.x + j] += XY[threadIdx.x * blockDim.x + j - 1];
	}
	__syncthreads();

	//##### DA MIGLIORARE UN ATTIMINO E' POSSIBILE USARE COME ACCUMULATORE L'IDICE DELL'ULTIMA CELLA DI OGNI SUBSECTION ########
	// phase 2: perform iterative kogge_stone_scan on XY
	float acc = 0;
	for (int stride = 1; stride <= threadIdx.x; stride++)
	{
		acc += XY[stride * blockDim.x - 1];
	}
	if (threadIdx.x > 0) {
		XY[threadIdx.x * blockDim.x + blockDim.x - 1] += acc;
	}
	__syncthreads();
	
	
	// phase 3: 
	for (unsigned int stride = 0; stride < blockDim.x - 1; stride++) {
			__syncthreads();
			XY[threadIdx.x * blockDim.x + blockDim.x + stride] += XY[threadIdx.x * blockDim.x + blockDim.x - 1];
		
	}
	__syncthreads();

	for (int j = 0; j < SECTION_SIZE; j += blockDim.x) {
		Y[threadIdx.x + j] = XY[threadIdx.x + j];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
	const int arraySize = 32;
	float *Y, *YS, *X;
	//float X[arraySize] = { 2,1,3,1,0,4,1,2,0,3,1,2,5,3,1,2 };

	X = (float*)malloc(arraySize * sizeof(float));
	Y = (float*)malloc(arraySize * sizeof(float));
	YS = (float*)malloc(arraySize * sizeof(float));

	//fill input vector
	
	for (int i = 0;	i < arraySize; i++ ) {
		X[i] = i + 1.0;
	}
	

	printf("Array input:");
	print_Array(X, arraySize);

	// perform sequential scan.
	printf("sequential scan:");
	sequential_scan(X, YS, arraySize);
	print_Array(YS, arraySize);
	//printf(" OK!\n");

	// perform parallel scan.
	printf("parallel scan:");
	cudaError_t cudaStatus = efficient_Kogge_Stone_scan(X, Y, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		goto Error;
	}
	print_Array(Y, arraySize);
	//printf(" OK!\n");

	// ------------------ verify the result. -----------------------------------
	if (verify_result(Y, YS, arraySize)) {
		goto Error;
	}
	printf("TEST PASSED!\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}

#ifdef WIN32
	system("pause");
#endif // WIN32
	return 0;

Error:
#ifdef WIN32
	system("pause");
#endif // WIN32
	return 1;
}

// Helper function for using CUDA to perform scan in parallel.
cudaError_t efficient_Kogge_Stone_scan(float *X, float *Y, unsigned int size)
{
	float *dev_X, *dev_Y;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors.
	cudaStatus = cudaMalloc((void**)&dev_X, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Y, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vector from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_X, X, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with 16 theads per block.
	dim3 dimGrid(ceil(size / 4.0), 1, 1);
	dim3 dimBlock(4.0, 1, 1);
	efficient_Kogge_Stone_scan_kernel << <dimGrid, dimBlock >> >(dev_X, dev_Y, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Y, dev_Y, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_X);
	cudaFree(dev_Y);

	return cudaStatus;
}

void sequential_scan(float *x, float *y, int Max_i) {
	float accumulator = x[0];
	y[0] = accumulator;
	for (int i = 1; i < Max_i; i++) {
		accumulator += x[i];
		y[i] = accumulator;
	}
}

void print_Array(float *A, int size) {
	for (int i = 0; i < size; i++) {
		printf("%.2f ", A[i]);
	}
	printf("\n\n");
}

int verify_result(float *Y, float *YS, int size) {
	for (int i = 0; i < size; i++) {
		if (Y[i] != YS[i]) {
			printf("Error Y[%d] = %.2f != %.2f = YS[%d]\n", i, Y[i], YS[i], i);
			return 1;
		}
	}
	return 0;
}
