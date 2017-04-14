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

__global__
void matrixVecMulKernel(float *d_Vout, float *d_M, float *d_V, int size) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread produce one output matrix element
	if (i < size && j < size) {
        float sum = 0;
        for (int k = 0; k < size; k++) {
            sum += d_M[i*size + k] * d_V[k];
        }
        d_Vout[i] = sum;
	}
}

// Compute Matrix-vector multiplcation h_Vout = h_M * h_V
void matrixVecMul(float *h_M, float *h_V, float *h_Vout, int size) {

    float *d_Vout, *d_M, d_V;

	CHECK_ERROR(cudaMalloc((void**)&d_Vout, size*sizeof(float)));
	CHECK_ERROR(cudaMalloc((void**)&d_M, size*size*sizeof(float)));
	CHECK_ERROR(cudaMalloc((void**)&d_V, size*sizeof(float)));

	cudaMemcpy(d_M, h_M, size*size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V, size*sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(size / 32.0), ceil(size / 32.0), 1);
	dim3 dimBlock(32.0, 32.0, 1);
	matrixVecMulKernel<<<dimGrid, dimBlock>>>(d_Vout, d_M, d_V, size);

	cudaMemcpy(h_Vout, d_Vout, size*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_Vout);
	cudaFree(d_M);
	cudaFree(d_V);
}

int main(void) {

	float *h_M, *h_V, *h_Vout;
	int size = 10; // assume square matrix

	h_M = (float*)malloc(sizeof(float)*size*size);
	h_V = (float*)malloc(sizeof(float)*size);
	h_Vout = (float*)malloc(sizeof(float)*size);

    printf("----- Matrix ------\n");
	// fill Min1 and Min2 with random float numbers
	srand(time(NULL));
	for (int i = 0; i < size ; i++) {
		for (int j = 0; j < size ; j++) {
			//h_M[i*size+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
            h_M[i*size+j] = (float)i*size+(j+1);
            printf("%2.2f ", h_M[i*size+j]);
			
		}
        //h_V[i] = ((((float)rand() / (float)(RAND_MAX)) * 10));
        h_V[i] = i+1;
        printf("\n");
	}
    printf("\n");
    
    printf("----- Vector ------\n");
    for (int i = 0; i < size; i++) {
       
        printf("%2.2f\n", h_V[i]);
    }
    printf("\n");

	// perform matrix vector multiplication
	matrixVecMul(h_M, h_V, h_Vout, size);

    
	// verify the result
	int valueIsCorrect = 1;
    
	for (int i = 0; i < size && valueIsCorrect; i++) {
        int sum = 0;
		for (int j = 0; j < size; j++) {
            sum += h_M[i*size + j] * h_V[i];
		}
        if (h_Vout[i] != sum) {
            printf("result is wrong!!!\n");
            valueIsCorrect = 0;
        }
        //printf("%2.2f\n", h_Vout[i]);
    }
	if (valueIsCorrect) {
		printf("matrixAdd complete with success!\n");
	}

	free(h_M);
	free(h_V);
	free(h_Vout);

	return 0;
}
