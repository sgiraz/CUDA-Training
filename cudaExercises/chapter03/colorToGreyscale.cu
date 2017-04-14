/*
 * EXAMPLE OF MAPPING THREADS TO MULTIDIMENSIONAL DATA: CHAPTER 3
 */

#include"lodepng.h"
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

#define CHANNELS 4


// input image is encoded as unsigned characters [0,255]
__global__
void colorToGrayscaleConversionKernel(unsigned char *Pin, unsigned char *Pout, int width, int height) {
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // check that only the threads with both Row and Col values are in within range
    if ( Col < width  && Row < height) {
        
        // get 1D coordine for the grayscale image
        int greyOffset = Row * width + Col;
        
        // one can think to RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset ];		// red value for pixel
        unsigned char g = Pin[rgbOffset + 1];	// green value for pixel
        unsigned char b = Pin[rgbOffset + 2];	// blue value for pixel
        
        // perform the rescaling and store it
        // we multiply by floating point constants
        Pout[rgbOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
        Pout[rgbOffset+1] = 0.21f*r + 0.71f*g + 0.07f*b;
        Pout[rgbOffset+2] = 0.21f*r + 0.71f*g + 0.07f*b;
        Pout[rgbOffset+3] = 255;
    }
}


void colorToGrayscaleConversion(unsigned char *h_Pin, unsigned char *h_Pout, int m, int n) {
    
    int size = (m*n*4)*sizeof(unsigned char);
    unsigned char *d_Pin, *d_Pout;
    
    //1. Allocate global memory on the device for d_Pin and d_Pout
    // With this type of allocation it isn't possible acces using higher-dimensional indexing syntax
    // it need to linearize first.
    CHECK_ERROR(cudaMalloc((void**)&d_Pin, size));
    CHECK_ERROR(cudaMalloc((void**)&d_Pout, size));
    
    // copy h_Pin to device memory
    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);
    
    //2. Kernel launch code - with 256 threads per block
    dim3 dimGrid(ceil(m / 16.0),ceil(n / 16.0),1);
    dim3 dimBlock(16, 16,1);
    colorToGrayscaleConversionKernel<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, m, n);
    
    //3. copy d_Pout from the device memory
    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);
    
    // Free device vectors
    cudaFree(d_Pin);
    cudaFree(d_Pout);
}


/*
 Decode from disk to raw pixels
 */
unsigned char* decodeOneStep(const char* filename)
{
    unsigned error;
    unsigned char* image;
    unsigned width, height;
    
    error = lodepng_decode32_file(&image, &width, &height, filename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
    
    return image;
}

/*
 Encode from raw pixels to disk with a single function call
 The image argument has width * height RGBA pixels or width * height * 4 bytes
 */
void encodeOneStep(const char* filename, unsigned char* image, int width, int height)
{
    /*Encode the image*/

    unsigned error = lodepng_encode32_file(filename, image, width, height);
    
    /*if there's an error, display it*/
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char *argv[]) {
    
    /* argv[1] must be the name of the image file */
    if (argc != 2) {
        printf("Usage: ./<executable_file> <image_file.png>\n");
        exit(1);
    }
    const char *filename = argv[1];
    
    // create host vectors
    unsigned char *h_Pin, *h_Pout;
    
    int m = 512; // track the pixel in x direction
    int n = 512; // track the pixel in y direction
    
    // allocate memory for host vectors
    h_Pin = (unsigned char*)malloc(sizeof(unsigned char)*(n*m));
    h_Pout = (unsigned char*)malloc(sizeof(unsigned char)*(n*m*4));
    
    
    // decode the .png image
    printf("decoding image...\n");
    h_Pin = decodeOneStep(filename);
    
    printf("colorToGrayscaleConversion...\n");
    //GpuTimer timer;
    //timer.Start();
    colorToGrayscaleConversion(h_Pin, h_Pout, m, n);
    //timer.Stop();
    
    printf("encoding converted image...\n");
    encodeOneStep("image_converted.png", h_Pout, m, n);
    
    printf("Ok conversion completed with success!\n");
    
    // Free host memory
    free(h_Pin);
    free(h_Pout);
    
    return 0;
}
