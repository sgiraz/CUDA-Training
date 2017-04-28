////////////////////////////////////////////////////////////////////////////
//
//  EXAMPLE OF HIERARCHICAL THREE PHASE PREFIX-SCAN CHAPTER 8
//  Efficient_Kogge_Stone_scan:
//	Using this three-phase approach, we can use a much smaller number of
//	threads then the number of the elements in a section. The maximal size
//	of a section is no longer limited by the number of threads in the block
//	but rather, the size of shared memory; all elements in a section
//	must to fit into the shared memory.
//
////////////////////////////////////////////////////////////////////////////
//	With 8192 elements using float numbers there are approximation problems 
////////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SECTION_SIZE 4096
#define BLOCK_DIM 1024
#define SUBSECTION_SIZE SECTION_SIZE / BLOCK_DIM

cudaError_t efficient_Kogge_Stone_scan(float *X, float *Y, unsigned int size, float *msTime);
void sequential_scan(float *x, float *y, int Max_i);
void print_Array(float *A, int size);
int verify_result(float *Y, float *YS, int size);

__device__
void print_Array_device(float *A, int size) {
	for (int i = 0; i < size; i++) {
		printf("A[%d] = %.2f\n", i, A[i]);
	}
	printf("\n\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Efficient prefix sum 
//! @param X  input data in global memory
//! @param Y  output data in global memory
//! @param InputSize size of input and output data
////////////////////////////////////////////////////////////////////////////////
__global__ void efficient_Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize) {
	__shared__ float XY[SECTION_SIZE];
	__shared__ float AUS[BLOCK_DIM];
	//int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Keep mind: Partition the input into blockDim.x subsections: i.e. for 8 threads --> 8 subsections

	// collaborative load in a coalesced manner
	for (int j = 0; j < SECTION_SIZE; j += blockDim.x) {
		XY[threadIdx.x + j] = X[threadIdx.x + j];
	}
	__syncthreads();


	// PHASE 1: scan inner own subsection
	// At the end of this phase the last element of each subsection contains the sum of all alements in own subsection
	for (int j = 1; j < SUBSECTION_SIZE; j++) {
		XY[threadIdx.x * (SUBSECTION_SIZE)+j] += XY[threadIdx.x * (SUBSECTION_SIZE)+j - 1];
	}
	__syncthreads();


	// PHASE 2: perform iterative kogge_stone_scan of the last elements of each subsections of XY loaded first in AUS
	AUS[threadIdx.x] = XY[threadIdx.x * (SUBSECTION_SIZE)+(SUBSECTION_SIZE)-1];
	float in;
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (threadIdx.x >= stride) {
			in = AUS[threadIdx.x - stride];
		}
		__syncthreads();
		if (threadIdx.x >= stride) {
			AUS[threadIdx.x] += in;
		}
	}
	__syncthreads();


	// PHASE 3: each thread adds to its elements the new value of the last element of its predecessor's section
	if (threadIdx.x > 0) {
		for (unsigned int stride = 0; stride < (SUBSECTION_SIZE)-1; stride++) {
			XY[threadIdx.x * (SUBSECTION_SIZE)+stride] += AUS[threadIdx.x - 1];  // <--
		}
	}
	__syncthreads();


	// store the result into output vector
	for (int j = 0; j < SECTION_SIZE; j += blockDim.x) {
		Y[threadIdx.x + j] = XY[threadIdx.x + j];
	}
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
	const int arraySize = 4096;
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

	//printf("Array input:");
	//print_Array(X, arraySize);

	// ---------------------- PERFORM SEQUENTIAL SCAN ----------------
	printf("Sequential scan...\n");
	cudaEventRecord(startTimeCuda, 0);
	cudaEventSynchronize(startTimeCuda);

	sequential_scan(X, YS, arraySize);

	cudaEventRecord(stopTimeCuda, 0);
	cudaEventSynchronize(stopTimeCuda);
	cudaEventElapsedTime(&msTime_seq, startTimeCuda, stopTimeCuda);
	printf("HostTime: %f\n\n", msTime_seq);
	//print_Array(YS, arraySize);
	//printf(" OK!\n");

	// ---------------------- PERFORM PARALELL SCAN ------------------
	printf("Parallel scan...\n");
	cudaError_t cudaStatus = efficient_Kogge_Stone_scan(X, Y, arraySize, &msTime);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		goto Error;
	}
	//print_Array(Y, arraySize);
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
cudaError_t efficient_Kogge_Stone_scan(float *X, float *Y, unsigned int size, float *msTime)
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

	// Launch a kernel on the GPU with BLOCK_DIM theads per block.
	cudaEventRecord(startTimeCuda, 0);
	cudaEventSynchronize(startTimeCuda);

	efficient_Kogge_Stone_scan_kernel << <1, BLOCK_DIM >> > (dev_X, dev_Y, size);

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
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching efficient_Kogge_Stone_scan_kernel Kernel!\n", cudaStatus);
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
		if (Y[i] - YS[i] > 1e-5) {
			printf("Error Y[%d] = %.2f != %.2f = YS[%d]\n", i, Y[i], YS[i], i);
			return 1;
		}
	}
	return 0;
}
