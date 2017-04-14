/*
 * EXAMPLE OF MAPPING THREADS TO MULTIDIMENSIONAL DATA (BLUR IMAGE): CHAPTER 3
 *
 * WORK IN PROGRES...
 *
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include"lodepng.h"

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

#define BLUR_SIZE 1 // 3x3 patch --> 2*BLUR_SIZE = number of pixels for each side of patch + 1
#define CHANNEL 4


__global__
void blurKernel(unsigned char *Pin, unsigned char *Pout, int width, int height) {
    
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // check that only the threads with both Row and Col values are in within range
    if ( Col < width  && Row < height) {
    	int pixVal = 0;
    	int pixels = 0;
        
    	// Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
    	// for each pixel of the patch
    	for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++){
            
    		for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; blurCol++){
                
    			int curRow = Row + blurRow;
    			int curCol = Col + blurCol;
    			
    			//verify we have a valid image pixel
    			if(curRow > -1 && curRow < height && curCol > -1 && curCol < width){
    				pixVal += Pin[curRow * width + curCol];
    				pixels++;
    			}
    		}
    	}
        //write our new pixel value out
        Pout[(Row * width + Col) * CHANNEL] = (unsigned char)(pixVal/pixels);
        Pout[(Row * width + Col) * CHANNEL + 1] = (unsigned char)(pixVal/pixels);
        Pout[(Row * width + Col) * CHANNEL + 2] = (unsigned char)(pixVal/pixels);
        Pout[(Row * width + Col) * CHANNEL + 3] = 255;
    }
}


void blur(unsigned char *h_Pin, unsigned char *h_Pout, int m, int n) {
    
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
    dim3 dimGrid(ceil(m / 16.0),ceil(n / 16.0), 1);
    dim3 dimBlock(16.0, 16.0, 1);
    blurKernel<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, m, n);
    
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
        printf("Usage: ./<executable_file>.x <name_of_image_file>\n");
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
    
    printf("blurConversion...\n");
    //GpuTimer timer;
    //timer.Start();
    blur(h_Pin, h_Pout, m, n);
    //timer.Stop();
    
    printf("encoding converted image...\n");
    encodeOneStep("blurImage.png", h_Pout, m, n);
    
    printf("ok conversion completed with success!\n");
    
    // Free host memory
    free(h_Pin);
    free(h_Pout);
    
    return 0;
}
