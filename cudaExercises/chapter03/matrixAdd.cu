/*
 * EXERCISE 1 CHAPTER 03  
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}


// each thread performs one pair-wise addition
// executed on the device, only callable from the host
__global__
void matrixAddKernel(float *d_Mout, float *d_Min1, float *d_Min2, int rows, int cols) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	/*
	// each thread produce one output matrix element (SOLUTION B)
	if (i < rows && j < cols) {
		d_Mout[i*rows + j] = d_Min1[i*rows + j] + d_Min2[i*rows + j];
	}
	*/

	/*
	// each thread produce one output matrix row (SOLUTION C)
	if (j == 0 && i < rows) {
		while (j < cols) {
			d_Mout[i*rows + j] = d_Min1[i*rows + j] + d_Min2[i*rows + j];
			j++;
		}
	}
	*/

	// each thread produce one output matrix column (SOLUTION D)
	if (i == 0 && j < cols) {
		while (i < rows) {
			d_Mout[i*rows + j] = d_Min1[i*rows + j] + d_Min2[i*rows + j];
			i++;
		}
	}
}

// Compute vector sum h_C = h_A + h_B
void matrixAdd(float *h_Mout, float *h_Min1, float *h_Min2, int rows, int cols) {

	float *d_Mout;
	float *d_Min1;
	float *d_Min2;
	int size = rows*cols *sizeof(float);

	CHECK_ERROR(cudaMalloc((void**)&d_Mout, size));
	CHECK_ERROR(cudaMalloc((void**)&d_Min1, size));
	CHECK_ERROR(cudaMalloc((void**)&d_Min2, size));

	cudaMemcpy(d_Min1, h_Min1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Min2, h_Min2, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(rows / 32.0), ceil(cols / 32.0), 1);
	dim3 dimBlock(32.0, 32.0, 1);

	matrixAddKernel<<<dimGrid, dimBlock>>>(d_Mout, d_Min1, d_Min2, rows, cols);

	cudaMemcpy(h_Mout, d_Mout, size, cudaMemcpyDeviceToHost);

	cudaFree(d_Mout);
	cudaFree(d_Min1);
	cudaFree(d_Min2);
}

int main(void) {

	float *h_Min1, *h_Min2, *h_Mout;
	int rows = 100;
	int cols = 100;

	h_Min1 = (float*)malloc(sizeof(float)*rows*cols);
	h_Min2 = (float*)malloc(sizeof(float)*rows*cols);
	h_Mout = (float*)malloc(sizeof(float)*rows*cols);

	// fill Min1 and Min2 with random float numbers
	srand(time(NULL));
	for (int i = 0; i < rows ; i++) {
		for (int j = 0; j < cols ; j++) {
			h_Min1[i*rows+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
			h_Min2[i*rows+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
		}
	}

	// perform matrix addiction
	matrixAdd(h_Mout, h_Min1, h_Min2, rows, cols);

	// verify the addition
	int valueIsCorrect = 1;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (h_Mout[i*rows + j] != (h_Min1[i*rows + j] + h_Min2[i*rows + j])) {
				printf("sum wrong!\n");
				valueIsCorrect = 0;
			}
		}
	}
	if (valueIsCorrect) {
		printf("matrixAdd complete with success!\n");
	}

	free(h_Min1);
	free(h_Min2);
	free(h_Mout);

	return 0;
}
