////////////////////////////////////////////////////////////////////////////
//
//  EXAMPLE OF INCLUSIVE PREFIX-SCAN CHAPTER 8
//  Brent_Kung_scan
//
////////////////////////////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//The maximum number of threads is of section elements
#define SECTION_SIZE 2048

cudaError_t Brent_Kung_scan(float *X, float *Y, unsigned int size, float *msTime);
void sequential_scan(float *x, float *y, int Max_i);
void print_Array(float *A, int size);
int verify_result(float *Y, float *YS, int size);

////////////////////////////////////////////////////////////////////////////////
//! Simple bad prefix sum 
//! @param X  input data in global memory
//! @param Y  output data in global memory
//! @param InputSize size of input and output data
////////////////////////////////////////////////////////////////////////////////
__global__ void Brent_Kung_scan_kernel(float *X, float *Y, int InputSize)
{
	__shared__ float XY[SECTION_SIZE];
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	if (i < InputSize) XY[threadIdx.x] = X[i];
	if (i + blockDim.x < InputSize) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];

	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < SECTION_SIZE) {
			XY[index] += XY[index - stride];
		}
	}

	for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < SECTION_SIZE) {
			XY[index + stride] += XY[index];
		}
	}

	__syncthreads();
	if (i < InputSize) Y[i] = XY[threadIdx.x];
	if (i + blockDim.x < InputSize) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
	const int arraySize = 2048;
	float *Y, *YS, *X;
	//float X[arraySize] = { 2,1,3,1,0,4,1,2,0,3,1,2,5,3,1,2 };
	float msTime, msTime_seq;
	cudaEvent_t startTimeCuda, stopTimeCuda;
	cudaEventCreate(&startTimeCuda);
	cudaEventCreate(&stopTimeCuda);

	X = (float*)malloc(arraySize * sizeof(float));
	Y = (float*)malloc(arraySize * sizeof(float));
	YS = (float*)malloc(arraySize * sizeof(float));

	//fill input vector
	for (int i = 0; i < arraySize; i++) {
		X[i] = (float)(i + 1.0);
	}

	//printf("Array input: ");
	//print_Array(X, arraySize);

	// ------------------ Perform sequential scan. -----------------------------
	printf("Sequential scan...\n");
	cudaEventRecord(startTimeCuda, 0);
	cudaEventSynchronize(startTimeCuda);

	sequential_scan(X, YS, arraySize);

	cudaEventRecord(stopTimeCuda, 0);
	cudaEventSynchronize(stopTimeCuda);
	cudaEventElapsedTime(&msTime_seq, startTimeCuda, stopTimeCuda);
	printf("HostTime: %f\n\n", msTime_seq);
	//printf(" OK!\n");
	//print_Array(YS, arraySize);

	// ------------------ perform parallel scan. -------------------------------
	printf("Parallel scan...\n");
	cudaError_t cudaStatus = Brent_Kung_scan(X, Y, arraySize, &msTime);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		goto Error;
	}
	//printf(" OK!\n");
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

	printf("Speedup: %f\n", msTime_seq / msTime);

	free(X);
	free(Y);
	free(YS);
#ifdef WIN32
	system("pause");
#endif // WIN32
	return 0;

Error:
	free(X);
	free(Y);
	free(YS);
#ifdef WIN32
	system("pause");
#endif // WIN32
	return 1;
}

// Helper function for using CUDA to perform scan in parallel.
cudaError_t Brent_Kung_scan(float *X, float *Y, unsigned int size, float *msTime)
{
	float *dev_X, *dev_Y;
	cudaError_t cudaStatus;
	cudaEvent_t startTimeCuda, stopTimeCuda;
	cudaEventCreate(&startTimeCuda);
	cudaEventCreate(&stopTimeCuda);

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
	cudaEventRecord(startTimeCuda, 0);
	cudaEventSynchronize(startTimeCuda);

	Brent_Kung_scan_kernel << < 1, SECTION_SIZE/2 >> >(dev_X, dev_Y, size);

	cudaEventRecord(stopTimeCuda, 0);
	cudaEventSynchronize(stopTimeCuda);
	cudaEventElapsedTime(msTime, startTimeCuda, stopTimeCuda);
	printf("KernelTime: %f\n\n", *msTime);

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
