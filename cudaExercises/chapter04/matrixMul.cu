/*
 * EXAMPLE OF MATRIX-MATRIX MULTIPLICATION: CHAPTER 4
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
void matrixMulKernel(float *P, float *M, float *N, int Width) {
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // check that only the threads with both Row and Col values are in within range
    if ( Row < Width && Col < Width ) {
        float pValue = 0;
        for(int k = 0; k < Width; k++){
            pValue += M[Row*Width+k] * N[k*Width+Col];
        }
        P[Row*Width+Col] = pValue;
    }
}


void matrixMul(float *h_P, float *h_M, float *h_N, int dim) {
    
    int size = (dim*dim)*sizeof(float);
    float *d_M, *d_N, *d_P;
    
    //1. Allocate global memory on the device for d_Pin and d_Pout
    // With this type of allocation it isn't possible acces using higher-dimensional indexing syntax
    // it need to linearize first.
    CHECK_ERROR(cudaMalloc((void**)&d_M, size));
    CHECK_ERROR(cudaMalloc((void**)&d_N, size));
    CHECK_ERROR(cudaMalloc((void**)&d_P, size));    // assume square matricies
    
    // copy h_Pin to device memory
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    
    //2. Kernel launch code - with 256 threads per block
    dim3 dimGrid(ceil(dim / 16.0),ceil(dim / 16.0),1);
    dim3 dimBlock(16, 16,1);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, dim);
    
    //3. copy d_Pout from the device memory
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    
    // Free device vectors
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char *argv[]) {
    
    float *h_M, *h_N, *h_P;
    int dim = 1024; // assume square matricies
    
    h_M = (float*)malloc(sizeof(float)*dim*dim);
    h_N = (float*)malloc(sizeof(float)*dim*dim);
    h_P = (float*)malloc(sizeof(float)*dim*dim);
    
    // fill M and N with random float numbers
    srand(time(NULL));
    for (int i = 0; i < dim ; i++) {
        for (int j = 0; j < dim ; j++) {
            h_M[i*dim+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
            h_N[i*dim+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
        }
    }
    
    // perform matrix addiction
    matrixMul(h_P, h_M, h_N, dim);
    
    /*********************************************************************************************************
     // verifiy the result
     int valueIsCorrect = 1;
     float mult[dim][dim];
     
     for (int i = 0; i < dim; i++) {
     for (int j = 0; j < dim; j++) {
     mult[i][j] = 0.0;
     }
     }
     
     // Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
     for(int i = 0; i < dim; ++i) {
     for(int j = 0; j < dim; ++j) {
     for(int k = 0; k < dim; ++k) {
     mult[i][j] += h_M[i*dim+k] * h_N[k*dim+j];
     }
     }
     }
     
     for (int i = 0; i < dim && valueIsCorrect; i++) {
     for (int j = 0; j < dim; j++) {
     printf("h_P[%d] != mult[%d][%d] --|-- %f != %f\n", (i*dim+j), i, j, h_P[i*dim+j], mult[i][j]);
     if (h_P[i*dim+j] != mult[i][j]) {
     valueIsCorrect = 0;
     printf("see error above.....\n");
     break;
     }
     }
     }
     ********************************************************************************************************
     * NON HA SENSO VERIFICARE LA CORRETTEZZA DEL RISULTATO SULL'HOST, VEDI 3.2 fino a 6.0 AL SEGUENTE LINK:
     * http://docs.nvidia.com/cuda/floating-point/
     ********************************************************************************************************/
    
    
    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);
    
    printf("ok multiplication completed with success!\n");
    
    /*
     if (valueIsCorrect) {
     printf("ok multiplication completed with success!\n");
     }
     else printf("somthing was wrong!\n");
     */
    
    return 0;
}
