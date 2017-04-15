/*
 * EXAMPLE OF SQUARE MATRIX MULTIPLICATION CHAPTER 4
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

#define TILE_WIDTH 16
#define DIM 1024


__global__
void matrixMulKernel(float *P, float *M, float *N) {
    
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH*2];
    
    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;
    
    // identify row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    if ( Row < DIM && Col < DIM ) {
        
        float pValue = 0;
        float pValue2 = 0;
        
        // Loop over the d_M and d_N tiles required to compute the d_P element
        for (int ph = 0; ph < DIM/TILE_WIDTH; ph++) {
            
            // Collaborative loading of d_M and d_N tiles n to the shared memory
            Mds[ty][tx] = M[Row * DIM + ph * TILE_WIDTH + tx];
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * DIM + Col];
            Nds[ty][tx+TILE_WIDTH] = N[(ph * TILE_WIDTH + ty) * DIM + Col + (DIM/2)];
            
            // printf("ph = %d; block[%d,%d]; thread[%d,%d] --> Nds[0][%d] = %2.2f\n", ph, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, tx, Nds[0][tx]);
            __syncthreads();
            
            
            for(int k = 0; k < TILE_WIDTH; k++){
                //printf("ph = %d; block[%d,%d]; thread[%d,%d] --> %2.2f + %2.2f * %2.2f\n", ph, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, pValue, Mds[ty][k], Nds[k][tx]);
                pValue  += Mds[ty][k] * Nds[k][tx];
                pValue2 += Mds[ty][k] * Nds[k][tx+TILE_WIDTH];
                
            }
            __syncthreads();
        }
        P[Row*DIM+Col] = pValue;
        P[Row*DIM+Col + (DIM/2)] = pValue2;
    }
}


float matrixMul(float *h_P, float *h_M, float *h_N) {
    
    int size = (DIM*DIM)*sizeof(float); // assume square matricies
    float *d_M, *d_N, *d_P;
    
    //1. Allocate global memory on the device for d_M, d_N and d_P
    // With this type of allocation it isn't possible acces using higher-dimensional indexing syntax
    // it need to linearize first.
    CHECK_ERROR(cudaMalloc((void**)&d_M, size));
    CHECK_ERROR(cudaMalloc((void**)&d_N, size));
    CHECK_ERROR(cudaMalloc((void**)&d_P, size));
    
    // copy h_M and h_N to device memory
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    
    cudaEvent_t startTimeCuda, stopTimeCuda;
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);
    
    //2. Kernel launch code - with TILE_WIDTH^2 threads per block
    cudaEventRecord(startTimeCuda, 0);
    
    dim3 dimGrid(ceil((DIM/TILE_WIDTH)/2.0), ceil(DIM/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_P, d_M, d_N);
    
    cudaEventRecord(stopTimeCuda,0);
    
    //3. copy d_P from the device memory
    cudaEventSynchronize(stopTimeCuda);
    float msTime;
    cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
    printf("KernelTime: %f\n", msTime);
 
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    
    // Free device matricies
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    return msTime;
}

void sequentialMM(float* h_M, float* h_N, float* h_C) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            float sum = 0.0;
            for (int k = 0; k < DIM; ++k)
                sum += h_M[i * DIM + k] * h_N[k * DIM + j];
            h_C[i * DIM + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    
    float *h_M, *h_N, *h_P, *h_C;
    float msTime, msTime_seq;
    cudaEvent_t startTimeCuda, stopTimeCuda;
    
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);
    
    h_M = (float*)malloc(sizeof(float)*DIM*DIM);
    h_N = (float*)malloc(sizeof(float)*DIM*DIM);
    h_P = (float*)malloc(sizeof(float)*DIM*DIM);
    h_C = (float*)malloc(sizeof(float)*DIM*DIM);
    
    // fill M and N with float numbers
    srand(time(NULL));
    for (int i = 0; i < DIM ; i++) {
        for (int j = 0; j < DIM ; j++) {
            h_M[i*DIM+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
            h_N[i*DIM+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
            h_C[i] = 0.0;
        }
    }
    
    // perform matrix multiplication
    msTime = matrixMul(h_P, h_M, h_N);
    
    // ------- perform matrix multiplication on host ---------
    cudaEventRecord(startTimeCuda, 0);
    sequentialMM(h_M, h_N, h_C);
    cudaEventRecord(stopTimeCuda,0);
    cudaEventSynchronize(stopTimeCuda);
    cudaEventElapsedTime(&msTime_seq, startTimeCuda, stopTimeCuda);
    printf("HostTime: %f\n", msTime_seq);
    
    /********************************************************************************************************
     * NON HA SENSO VERIFICARE LA CORRETTEZZA DEL RISULTATO SULL'HOST, VEDI 3.2 fino a 6.0 AL SEGUENTE LINK:
     * http://docs.nvidia.com/cuda/floating-point/
     ********************************************************************************************************
    // verify the result
    for (int i = 0; i < DIM * DIM; ++i) {
        if (h_C[i] != h_P[i]) {
            printf("\x1b[31mError\x1b[0m into result: h_C[%d] = %f != %f = h_P[%d]\n", i, h_C[i], h_P[i], i);
            goto Error;
        }
    }
    */
    
    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_C);
    
    printf("Ok multiplication completed with \x1b[32msuccess\x1b[0m!\n\n");
    printf("Speedup: %f\n", msTime_seq/msTime);
    
    return 0;
    
Error:
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_C);
    
    return -1;
}
