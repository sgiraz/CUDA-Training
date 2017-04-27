////////////////////////////////////////////////////////////////////////////
//
//  EXAMPLE OF BAD EXLUSIVE PREFIX-SCAN CHAPTER 8
//  exclusive Kogge_Stone_scan
//
////////////////////////////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Assumption: the number of threads will be equal to section elements
#define SECTION_SIZE 32

cudaError_t Kogge_Stone_scan(float *X, float *Y, unsigned int size);
void sequential_scan(float *x, float *y, int Max_i);
void print_Array(float *A, int size);
int verify_result(float *Y, float *YS, int size);

////////////////////////////////////////////////////////////////////////////////
//! Simple bad prefix sum 
//! @param X  input data in global memory
//! @param Y  output data in global memory
//! @param InputSize size of input and output data
////////////////////////////////////////////////////////////////////////////////
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize)
{
	__shared__ float XY[SECTION_SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < InputSize && threadIdx.x != 0) {
		XY[threadIdx.x] = X[i - 1];
	}
	else {
		XY[threadIdx.x] = 0;
	}

	if (threadIdx.x < InputSize)
	{
		// Perform iterative exclusive scan on XY
		for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
			if (threadIdx.x >= stride) {
				__syncthreads();
				XY[threadIdx.x] += XY[threadIdx.x - stride];
			}
		}
		Y[i] = XY[threadIdx.x];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
	const int arraySize = 8;
	float X[arraySize] = { 3, 1, 7, 0, 4, 1, 6, 3 };
	float Y[arraySize];
	float YS[arraySize];


	//printf("Array input: ");
	//print_Array(X, arraySize);

	// ------------------ Perform sequential scan. -----------------------------
	printf("Sequential scan...");
	sequential_scan(X, YS, arraySize);
	printf(" OK!\n");
	//print_Array(YS, arraySize);

	// ------------------ perform parallel scan. -------------------------------
	printf("Parallel scan...");
	cudaError_t cudaStatus = Kogge_Stone_scan(X, Y, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		goto Error;
	}
	printf(" OK!\n");
	//print_Array(Y, arraySize);

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
cudaError_t Kogge_Stone_scan(float *X, float *Y, unsigned int size)
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

	// Launch a kernel on the GPU with one thread for each element.
	dim3 dimGrid(ceil(size / float(SECTION_SIZE)), 1, 1);
	dim3 dimBlock(SECTION_SIZE, 1, 1);
	Kogge_Stone_scan_kernel << <dimGrid, dimBlock >> >(dev_X, dev_Y, size);

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
	float accumulator = 0;
	y[0] = accumulator;
	for (int i = 1; i < Max_i; i++) {
		accumulator += x[i-1];
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
	for (int i = 0; i < size ; i++) {
		if (Y[i]!=YS[i]) {
			printf("Error Y[%d] = %.2f != %.2f = YS[%d]\n", i, Y[i], YS[i], i);
			return 1;
		}
	}
	return 0;
}
