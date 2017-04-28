////////////////////////////////////////////////////////////////////////////
//
//  EXAMPLE OF HIERARCHICAL THREE PHASE PREFIX-SCAN CHAPTER 8
//  kernel_1 - three-phase-scan
//  kernel_2 - Kogge stone-scan
//  kernel_3 - takes the S and Y arrays as input and writes its output back into Y
//
////////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Here we assume that INPUT_SIZE is a multiple of SECTION_SIZE
#define INPUT_SIZE 2097152
#define SECTION_SIZE 2048
#define BLOCK_DIM (SECTION_SIZE/2)
#define SUBSECTION_SIZE (SECTION_SIZE/BLOCK_DIM)

cudaError_t efficient_Kogge_Stone_scan(int *X, int *Y, int *S, float *msTime);
void sequential_scan(int *x, int *y, int Max_i);
void print_Array(int *A, int size);
int verify_result(int *Y, int *YS, int size);


////////////////////////////////////////////////////////////////////////////////
//! First kernel for hierarchical scan --- three-phase-scan
//! @param X  input data in global memory
//! @param Y  output data in global memory
//! @param S  support vector for last element of each section
////////////////////////////////////////////////////////////////////////////////
__global__ void hierarchical_scan_kernel_phase1(int *X, int *Y, int *S) {
	__shared__ int XY[SECTION_SIZE];
	__shared__ int AUS[BLOCK_DIM];
	int tx = threadIdx.x, bx = blockIdx.x;
	int i = bx * SECTION_SIZE + tx;

	if (i < INPUT_SIZE) {

		// collaborative load in a coalesced manner
		for (int j = 0; j < SECTION_SIZE; j+=BLOCK_DIM) {
			XY[tx + j] = X[i + j];
		}
		__syncthreads();

		
		// PHASE 1: scan inner own subsection
		// At the end of this phase the last element of each subsection contains the sum of all alements in own subsection
		for (int j = 1; j < SUBSECTION_SIZE; j++) {
			XY[tx * (SUBSECTION_SIZE) + j] += XY[tx * (SUBSECTION_SIZE)+j - 1];
		}
		__syncthreads();


		// PHASE 2: perform iterative kogge_stone_scan of the last elements of each subsections of XY loaded first in AUS
		AUS[tx] = XY[tx * (SUBSECTION_SIZE)+(SUBSECTION_SIZE)-1];
		int in;
		for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2) {
			__syncthreads();
			if (tx >= stride) {
				in = AUS[tx - stride];	
			}
			__syncthreads();
			if (tx >= stride) {
				AUS[tx] += in;
			}
		}
		__syncthreads();

		// PHASE 3: each thread adds to its elements the new value of the last element of its predecessor's section
		if (tx > 0) {
			for (unsigned int stride = 0; stride < (SUBSECTION_SIZE); stride++) {
				XY[tx * (SUBSECTION_SIZE)+stride] += AUS[tx - 1];  // <--
			}
		}
		__syncthreads();
		
		// store the result into output vector
		for (int j = 0; j < SECTION_SIZE; j += BLOCK_DIM) {
			Y[i + j] = XY[tx + j];
		}

		//The last thread in the block writes the output value of the last element in the scan block to the blockIdx.x position of S
		if (tx == BLOCK_DIM - 1) {
			S[bx] = XY[SECTION_SIZE - 1];
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Second kernel for hierarchical scan --- Kogge-Stone_scan
//! @param S  input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void hierarchical_scan_kernel_phase2(int *S) {

	__shared__ int XY[SECTION_SIZE];
	int tx = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tx;

	if (i < INPUT_SIZE) {
		XY[tx] = S[i];
	}

	// Perform iterative scan on XY
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		int in;
		__syncthreads();
		if (tx >= stride) {
			in = XY[tx - stride];
		}
		__syncthreads();
		if (tx >= stride) {
			XY[tx] += in;
		}
	}

	S[i] = XY[tx];
}


////////////////////////////////////////////////////////////////////////////////
//! Third kernel for hierarchical scan
//! @param S  input data in global memory
//! @param Y  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void hierarchical_scan_kernel_phase3(int *S, int *Y) {

	int tx = threadIdx.x, bx = blockIdx.x;
	int i = bx * SECTION_SIZE + tx;
	//printf("Y[%d] = %.2f\n", i, Y[i]);
	
	if (bx > 0)
	{
		for (int j = 0; j < SECTION_SIZE ; j += BLOCK_DIM ) {
			Y[i + j] += S[bx - 1];
		}
	}
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
	int *Y, *YS, *S, *X;
	//int X[INPUT_SIZE] = { 2,1,3,1,0,4,1,2,0,3,1,2,5,3,1,2 };
	float msTime, msTime_seq;
	cudaEvent_t startTimeCuda, stopTimeCuda;
	cudaEventCreate(&startTimeCuda);
	cudaEventCreate(&stopTimeCuda);

	X = (int*)malloc(INPUT_SIZE * sizeof(int));
	Y = (int*)malloc(INPUT_SIZE * sizeof(int));
	YS = (int*)malloc(INPUT_SIZE * sizeof(int));
	S = (int*)malloc((INPUT_SIZE/SECTION_SIZE) * sizeof(int));

	//fill input vector
	for (int i = 0; i < INPUT_SIZE; i++) {
		X[i] = (int)(i + 1.0);
	}

	//printf("Array input:");
	//print_Array(X, INPUT_SIZE);

	// ---------------------- PERFORM SEQUENTIAL SCAN ----------------
	printf("Sequential scan...\n");
	cudaEventRecord(startTimeCuda, 0);
	cudaEventSynchronize(startTimeCuda);

	sequential_scan(X, YS, INPUT_SIZE);

	cudaEventRecord(stopTimeCuda, 0);
	cudaEventSynchronize(stopTimeCuda);
	cudaEventElapsedTime(&msTime_seq, startTimeCuda, stopTimeCuda);
	printf("HostTime: %f\n\n", msTime_seq);
	//print_Array(YS, INPUT_SIZE);
	//printf(" OK!\n");

	// ---------------------- PERFORM PARALELL SCAN ------------------
	printf("Parallel scan...\n");
	cudaError_t cudaStatus = efficient_Kogge_Stone_scan(X, Y, S, &msTime);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		goto Error;
	}
	//print_Array(Y, INPUT_SIZE);
	//printf(" OK!\n");

	// ---------------------- VERIFY THE RESULT ----------------------
	if (verify_result(Y, YS, INPUT_SIZE)) {
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
cudaError_t efficient_Kogge_Stone_scan(int *X, int *Y, int *S, float *msTime)
{
	int *dev_X, *dev_Y, *dev_S;
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

	// Allocate GPU buffers for three vectors.
	cudaStatus = cudaMalloc((void**)&dev_X, INPUT_SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Y, INPUT_SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_S, (INPUT_SIZE/SECTION_SIZE) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vector from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_X, X, INPUT_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with BLOCK_DIM theads per block.
	cudaEventRecord(startTimeCuda, 0);
	cudaEventSynchronize(startTimeCuda);

	hierarchical_scan_kernel_phase1 << < ceil(INPUT_SIZE / (float)SECTION_SIZE), BLOCK_DIM >> > (dev_X, dev_Y, dev_S);
	hierarchical_scan_kernel_phase2 << < 1, ceil(INPUT_SIZE / SECTION_SIZE) >> >(dev_S);
	hierarchical_scan_kernel_phase3 << < ceil(INPUT_SIZE / (float)SECTION_SIZE), BLOCK_DIM >> >(dev_S, dev_Y);

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
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching hierarchical_scan_kernel_phase1 Kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Y, dev_Y, INPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_X);
	cudaFree(dev_Y);

	return cudaStatus;
}

void sequential_scan(int *x, int *y, int Max_i) {
	int accumulator = x[0];
	y[0] = accumulator;
	for (int i = 1; i < Max_i; i++) {
		accumulator += x[i];
		y[i] = accumulator;
	}
}

void print_Array(int *A, int size) {
	for (int i = 0; i < size; i++) {
		printf("%.d ", A[i]);
	}
	printf("\n\n");
}

int verify_result(int *Y, int *YS, int size) {
	for (int i = 0; i < size; i++) {
		if (Y[i] != YS[i]) {
			printf("Error Y[%d] = %d != %d = YS[%d]\n", i, Y[i], YS[i], i);
			return 1;
		}
	}
	return 0;
}
