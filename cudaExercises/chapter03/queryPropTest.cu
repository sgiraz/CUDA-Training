/*
 * EXERCISE 1 CHAPTER 03  
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); EXIT_FAILURE; \
	} \
}

int main(void) {

    int dev_count;
    
    cudaGetDeviceCount(&dev_count);
    printf("Number of available CUDA device:%3d\n", dev_count);
    
    for (int i =  0; i < dev_count; i++) {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, i);
        
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", dev_prop.name);
        printf("  Compute capability: %d.%d\n", dev_prop.major, dev_prop.minor);
        printf("  Memory Clock Rate (KHz): %d\n", dev_prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", dev_prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*dev_prop.memoryClockRate*(dev_prop.memoryBusWidth/8)/1.0e6);
        printf("  Max thread per block: %d\n", dev_prop.maxThreadsPerBlock);
    }
    
	
	return 0;
}
