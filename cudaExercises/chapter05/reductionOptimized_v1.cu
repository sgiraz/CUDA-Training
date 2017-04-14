#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>


#define BLOCK_SIZE 16
cudaError_t addWithCuda(int *h_X, int size);


__global__ void partialSumKernel(int *X, int N)
{
	__shared__ int partialSum[2 * BLOCK_SIZE];
	int tx = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tx;
	partialSum[tx] = (i < N) ?  X[i] : 0;
	partialSum[tx + blockDim.x] = 0;

	for (int stride = blockDim.x; stride > 0; stride = stride/2)
	{
		__syncthreads();
		if (tx <= stride) {
			partialSum[tx] += partialSum[tx + stride];
			//printf("tx[%d], bx[%d]: %d + %d\n", tx, blockIdx.x, partialSum[tx], partialSum[tx + stride]);
		}
	}
	if (tx == 0)
		X[blockIdx.x] = partialSum[tx];
}

int main()
{
	int *h_X;
	int size = 32;

	h_X = (int*)malloc(sizeof(int)*size);
	
	// fill the vector with a simple for loop
	for (int i = 0; i < size; i++) {
		h_X[i] = i;
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(h_X, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
	for (int i = 0; i < ceil(((float)size)/BLOCK_SIZE) ; i++ ) {
		printf("the partial sum result in block %d is: %d\n", i, h_X[i]);
	}
	
	free(h_X);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *h_X, int size)
{
    int *d_X;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers 
    cudaStatus = cudaMalloc((void**)&d_X, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

     // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_X, h_X, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    partialSumKernel<<<ceil(((float)size) / BLOCK_SIZE), BLOCK_SIZE>>>(d_X, size);

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
    cudaStatus = cudaMemcpy(h_X, d_X, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaFree(d_X);

Error:
    cudaFree(d_X);

    return cudaStatus;
}
