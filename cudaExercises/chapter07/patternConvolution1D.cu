/*
 * EXAMPLE OF PATTERN CONVOLUTION CHAPTER 7
 */

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); EXIT_FAILURE; \
	} \
}


/*
// compute vector sum C = A+B
// each thread performs one pair-wise addition
__global__ // executed on the device, only callable from the host
void vecAddKernel(int *A, int *B, int *C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	// questo if controlla che il thread i-esimo acceda ad una zona di memoria valida per
	// gli array che sto considerando (ovvero tiene conto del caso in cui vengono generati thread in eccesso)
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

void vecAdd(int *h_M, int *h_N, int *h_P, int n) {

	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;

	//1. Allocate global memory on the device for A, B and C
	CHECK_ERROR(cudaMalloc((void**)&d_A, size));
	CHECK_ERROR(cudaMalloc((void**)&d_B, size));
	CHECK_ERROR(cudaMalloc((void**)&d_C, size));

	// copy A and B to device memory
	cudaMemcpy(d_A, h_M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_N, size, cudaMemcpyHostToDevice);

	//2. Kernel launch code - to have the device to perform the actual vector addition
	// Kernel invocation with 256 threads
	dim3 dimGrid = (ceil(n / 256.0),1,1);
	dim3 dimBlock((256.0),1,1);
	vecAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

	//3. copy C from the device memory
	cudaMemcpy(h_P, d_C, size, cudaMemcpyDeviceToHost);

	// Free device vectors
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
 */

void printArray(float *A, int size){
    for (int i = 0; i < size; i++) {
        printf("%.2f ", A[i]);
    }
    printf("\n");
}

int main(void) {

	// create and host vectors
    float *h_P;//, *h_N;
	const int n = 5;
    const int sizeMask = 3;
    const float val = (float)1/(float)2;
    printf("-val = %.2f\n\n", -val);
    
    float h_M[] = {-val,0,val};
    float h_N[] = {1.0,2.0,3.0,4.0,5.0};

	// allocate memory for host vectors
	//h_M = (int*)malloc(sizeof(int)*sizeMask);   // mask array
	//h_N = (float*)malloc(sizeof(float)*n);          // input array
	h_P = (float*)malloc(sizeof(float)*n);          // output array
	
	
	srand(time(NULL));
	for (int i = 0; i < n; i++) {
        h_P[i] = 0.0;
        //h_N[i] = ((float)rand() / (float)(RAND_MAX)) * 100;
	}

	// call vecAdd to compute vector sum
	//vecAdd(h_M, h_N, h_P, n);

    
	// sequential pattern convolution
    int pos;
	for (int i = 0; i < n; i++) {
        pos = sizeMask/2 - i;
        printf("P[%d] = ", i);
        for (int j = 0; j < sizeMask; j++) {
            
            if (j - pos < 0 || j - pos > n - 1) {
                h_P[i] +=  0 * h_M[j];
                printf("0 * %.2f %c ", h_M[j], (j - pos > n - 1) ? ' ' : '+' );
            }
            else{
                h_P[i] += h_N[j - pos] * h_M[j];
                printf("%.2f * %.2f %c ", h_N[j - pos], h_M[j], (j < sizeMask - 1) ? '+' : ' ' );
            }
        }
        printf("= %.2f\n", h_P[i]);
	}
    
    printf("---------- ARRAY INPUT ----------\n");
    printArray(h_N, n);
    
    printf("---------- ARRAY RESULT----------\n");
    printArray(h_P, n);

	// Free host memory
	//free(h_M);
	//free(h_N);
	//free(h_P);


	return 0;
}
