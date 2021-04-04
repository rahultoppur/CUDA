// INSTRUCTIONS HOW TO RUN
//
// (Run this from the root directory of this folder)
// Compile with: nvcc ./src/filter.cu ./../common/src/ppm.cu -I./../common/include -o filter
// Run with    : ./filter
//               You should then output ('display output.ppm')
//               to see your results.

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"    // Note this is found in the /common folder, so we include
                    // that filepath when we compile our source code as a 
                    // directory to find files such as "ppm.h" in.
#define MSG_LEN 513


// Here we have a function that runs on the GPU
// or otherwise, this is our 'device' code.
// This function will take in an array of pixels with
// R, G, and B components and multiply their value
// to create a 'grayscale' filter of an image.
__global__
void decode(unsigned char* pixel_array, int width, int height, char* msg) {

    // Here we are determining the (x,y) coordinate within
    // the 2D image.
    int x   = threadIdx.x + (blockDim.x*blockIdx.x);
    int y   = threadIdx.y + (blockDim.y*blockIdx.y);
  

    // Check to make sure we do not have more threads
    // than the index of our array
    if(x < width && y < height){
        
        // We then compute which exact pixel we
        // are located at.
        // 'y' selects the row, and 'x' selects
        // the column. The 'width' is the 'pitch'
        // of the image, and we multiply it by 'y'
        // to select which row in the image we are on.
        int offset = (y*width+x);
        unsigned char r, g, b;


        // We need to read the first 177 pixels and find the right-most bit
        if (offset < MSG_LEN) {
            unsigned char secret_bit = pixel_array[offset*3] & 1;
            //printf("Offset: %d, secret bit is: %d\n", offset, secret_bit);
            msg[offset] = secret_bit;

            /*
            if (msg[offset] == '0') {
                r = pixel_array[offset*3] & 0376;        
                printf("msg[%03d] = %c. R is: %d. R was %d.\n", offset, msg[offset], r, pixel_array[offset*3]);
            }
            else {
                r = pixel_array[offset*3] | 01;
                printf("msg[%03d] = %c. R is: %d. R was %d.\n", offset, msg[offset], r, pixel_array[offset*3]);
            }
            */
        }
        

        // Our pixels is then made up of 3 color componetns
        // R,G, and B. We multiply by '3' because once we
        // select our pixel, we actually have 3 values per pixel.
        // We then select the R,G, and B values by incrementing
        // by +0, +1, or +2 for R,G, and B.

       
        //unsigned char r = pixel_array[offset*3]; 
        //r = pixel_array[offset*3];
       // g = pixel_array[offset*3+1]; 
       // b = pixel_array[offset*3+2];

        
        //if(offset < 177) {        
        //    printf("%d,%d,%d\n", r, g, b);
        //}

       /* 
        pixel_array[offset*3] = r;
        pixel_array[offset*3+1] = g;
        pixel_array[offset*3+2] = b;
        */
        

       
    }
}


int main(int argc, char** argv){

    const char* input = "safari-encoded.ppm"; 
    ppm_t* myImage = loadPPMImage(input);
    // =================== CPU Code =========================

    // =================== GPU Code ========================= 
    // Here we are going to launch our kernel:
    // Now let's allocate some memory in CUDA for our pixels array
    // We need to allocate enough memory for all of our pixels, as well
    // as each of the R,G,B color components
    int numberOfPixels = myImage->width * myImage->height;
    //printf("Width: %d", myImage->width);
    int sizeOfDataInBytes = numberOfPixels*3*sizeof(unsigned char);
    // We next create a pointer which will point to where our
    // data has been allocated on the GPU.
    unsigned char* gpu_pixels_array; 
    char* gpu_msg;
    char msg[MSG_LEN+1];

    //char msg[] = "01110100011010000110010100100000011010110110111001101001011001110110100001110100011100110010000001110111011010000110111100100000011100110110000101111001001000000110111001101001";


    // Now we need to allocate a contigous block of memory
    // on the GPU that our pointer will point to.
    // We do this with the cudaMalloc function.
    cudaMalloc(&gpu_pixels_array,sizeOfDataInBytes);
    cudaMalloc(&gpu_msg,MSG_LEN+1);
    // We then copy from our 'host' the data from our CPU into our GPU.
    // Again, what this function is doing, is making a memory transfer of CPU
    // to memory that we have allocated on our GPU.
    cudaMemcpy(gpu_msg,msg,MSG_LEN+1, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pixels_array,myImage->pixels,sizeOfDataInBytes,cudaMemcpyHostToDevice);      
    //  Here are the parameters
    //    <<<number of thread blocks, number of threads per block>>>
    // So in total, we are launching a total number of threads
    // equal to: "number of thread blocks" * "number of threads per block"        
    // NOTE:    Be careful here with the first parameter of which array you are
    //          passing as the first parameter. Remember that CUDA functions
    //          that are executing on a 'device' should only address pointers
    //          and memory locations that are on the actual device, thus we
    //          pass in our gpu_pixels_array where we copied all of our pixels
    //          to in the previous step.
    dim3 dimGrid(myImage->width*3,myImage->height*3,1);
    dim3 dimBlock(1,1,1);
    // Now we call our grayscale function
    // Note that we are passing in our gpu_pixels_array in, as when we
    // work with GPU kernel code, we need to work with GPU memory
    // (i.e. blocks of memory allocated with cudaMalloc).
    decode<<<dimGrid,dimBlock>>>(gpu_pixels_array,myImage->width,myImage->height,gpu_msg);
    // After our kernel has been called, we copy the data from our GPU memory
    // back onto our CPU memory.
    // Our goal is to get the memory back on the CPU, so we can use
    // CPU functions to write our file back to disk.
    cudaMemcpy(myImage->pixels,gpu_pixels_array,sizeOfDataInBytes,cudaMemcpyDeviceToHost);     
    cudaMemcpy(msg, gpu_msg, MSG_LEN+1, cudaMemcpyDeviceToHost);
    // Free memory that we have allocated on the GPU.
    cudaFree(gpu_pixels_array);
    // =================== GPU Code =========================
    
     
    // =================== CPU Code =========================
    // Write our file back to disk
    //const char* output = "safari-encoded.ppm";
    //savePPMImage(output, myImage);

    // Free memory that we have allocated on the CPU.
    free(myImage);

    /* Display the secret message */
    for(int i=0; i < MSG_LEN; ++i) {
        printf("%d", msg[i]);
    }
    printf("\n");

    // =================== CPU Code =========================
    

    return 0;
}
