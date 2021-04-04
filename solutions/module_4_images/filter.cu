// INSTRUCTIONS HOW TO RUN
//
// (Run this from the root directory of this folder)
// Compile with: nvcc ./src/filter.cu ./../common/src/ppm.cu -I./../common/include -o filter
// Run with    : ./filter
//               You should then output ('display output.ppm')
//               to see your results.

#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"    // Note this is found in the /common folder, so we include
                    // that filepath when we compile our source code as a 
                    // directory to find files such as "ppm.h" in.

// Here we have a function that runs on the GPU
// or otherwise, this is our 'device' code.
// This function will take in an array of pixels with
// R, G, and B components and multiply their value
// to create a 'grayscale' filter of an image.
__global__
void make_grayscale(unsigned char* pixel_array, int width, int height){
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

        // Our pixels is then made up of 3 color componetns
        // R,G, and B. We multiply by '3' because once we
        // select our pixel, we actually have 3 values per pixel.
        // We then select the R,G, and B values by incrementing
        // by +0, +1, or +2 for R,G, and B.

       
        unsigned char r = pixel_array[offset*3]; 
        unsigned char g = pixel_array[offset*3+1]; 
        unsigned char b = pixel_array[offset*3+2];
        

        // At this step we now create our grayscale image.
        // We compute a dot product of the pixels, and apply
        // it to each of our R,G, and B components. Because
        // this is a grayscale image, all of the colors are
        // the same value.
        // NOTE: We could compress our image and only have 1
        //       color channel since all the values are the same.
        /*
        pixel_array[offset*3]       = r*.21f + g*.71f + b*.07f;
        pixel_array[offset*3+1]     = r*.21f + g*.71f + b*.07f;
        pixel_array[offset*3+2]     = r*.21f + g*.71f + b*.07f;
        */

       
       /* 
        unsigned char tr = r*0.21f + g*0.71f + b*0.07f;
        unsigned char tg = r*0.21f + g*0.71f + b*0.07f;
        unsigned char tb = r*0.21f + g*0.71f + b*0.07f;
      */ 


        /*
        unsigned char tr = (r*.393f + g*.769f + b*.189f);
        unsigned char tg = (r*.349f + g*.686f + b*.168f);
        unsigned char tb = (r*.272f + g*.534f + b*.131f);
         */      

        /*
        unsigned char tr = (r*.393f + g*.349f + b*.272f);
        unsigned char tg = (r*.769f + g*.686f + b*.534f);
        unsigned char tb = (r*.189f + g*.168f + b*.131f);
        */

       /* 
        pixel_array[offset*3] = (r*.393f + g*.769f + b*.189f);
        pixel_array[offset*3+1] = (r*.349f + g*.686f + b*.168f);
        pixel_array[offset*3+2] = (r*.272f + g*.534f + b*.131f);
      */ 

        /*
        pixel_array[offset*3+2] = (r*.393f + g*.349f + b*.272f);
        pixel_array[offset*3+1] = (r*.769f + g*.686f + b*.534f);
        pixel_array[offset*3] = (r*.189f + g*.168f + b*.131f);
        */

       
        unsigned char tr = g*.2f + b*.769f + r*.189f; 
        //unsigned char tr = r*.393f + b*.769f + g*.189f; // this is a blue tint

        unsigned char tg = .49f * tr;
        unsigned char tb = .69f * tr; 
       

        /*
        unsigned int tr = r-20;
        unsigned int tg = g+20;
        unsigned int tb = b+20; 
        */


        if(tr >= 255)
            tr = 255;
        if(tg >= 255)
            tg = 255;
        if(tb >= 255)
            tb = 255;

        //printf("%u,%u,%u\n", tr, tg, tb); 
        /*
        pixel_array[offset*3] = r*.393f + g*.769f + b*.189f;
        pixel_array[offset*3+1] = 0.89f*pixel_array[offset*3];
        pixel_array[offset*3+2] = 0.69f*pixel_array[offset*3];
        */


       /* 
        pixel_array[offset*3] = (unsigned char)tr;
        pixel_array[offset*3+1] = (unsigned char)tg;
        pixel_array[offset*3+2] = (unsigned char)tb;
       */ 

        //pixel_array[offset*3+2] = 0.69f*pixel_array[offset*3];


       /* 
        unsigned char tr = r;
        unsigned char tg = g;
        unsigned char tb = b;

        */
        /*
        r = r*.393f + g*.769f + b*.189f;
        g = r*.349f + g*.686f + b*.168f;
        b = r*.272f + g*.534f + b*.131f;
        */
 
       /* 
        r = (tr > 255) ? 255: tr;
        g = (tg > 255) ? 255: tg;
        b = (tb > 255) ? 255: tb;
        */

        
        pixel_array[offset*3] = tr;;
        pixel_array[offset*3+1] = tg;
        pixel_array[offset*3+2] = tb;
        
    }
}

// Here is another function for fun that 'brightens' the
// actual image
// For fun, you make explore this function as well.
__global__
void make_brighter(unsigned char* pixel_array, int size){
    int id  = threadIdx.x+blockDim.x*blockIdx.x;
 
    // Check to make sure we do not have more threads
    // than the index of our array
    if(id < size){
        pixel_array[id] *= 6; 
    }
}


int main(int argc, char** argv){

    // =================== CPU Code =========================
    // Our CPU must be used to perform I/O (i.e. input and
    // output from the disk). So the first thing we do is
    // load the data from our CPU onto our machine.
    // Our goal here is to load an image in a format that
    // we understand, and retrieve the raw the pixel color
    // information. Then we can manipulate the pixel data
    // to apply a filter for our image.
    const char* input = "safari.ppm"; 
    ppm_t* myImage = loadPPMImage(input);
    // =================== CPU Code =========================

    // =================== GPU Code ========================= 
    // Here we are going to launch our kernel:
    // Now let's allocate some memory in CUDA for our pixels array
    // We need to allocate enough memory for all of our pixels, as well
    // as each of the R,G,B color components
    int numberOfPixels = myImage->width * myImage->height;
    printf("Width: %d", myImage->width);
    int sizeOfDataInBytes = numberOfPixels*3*sizeof(unsigned char);
    // We next create a pointer which will point to where our
    // data has been allocated on the GPU.
    unsigned char* gpu_pixels_array; 
    // Now we need to allocate a contigous block of memory
    // on the GPU that our pointer will point to.
    // We do this with the cudaMalloc function.
    cudaMalloc(&gpu_pixels_array,sizeOfDataInBytes);
    // We then copy from our 'host' the data from our CPU into our GPU.
    // Again, what this function is doing, is making a memory transfer of CPU
    // to memory that we have allocated on our GPU.
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
    make_grayscale<<<dimGrid,dimBlock>>>(gpu_pixels_array,myImage->width,myImage->height);
    // After our kernel has been called, we copy the data from our GPU memory
    // back onto our CPU memory.
    // Our goal is to get the memory back on the CPU, so we can use
    // CPU functions to write our file back to disk.
    cudaMemcpy(myImage->pixels,gpu_pixels_array,sizeOfDataInBytes,cudaMemcpyDeviceToHost);     
    // Free memory that we have allocated on the GPU.
    cudaFree(gpu_pixels_array);
    // =================== GPU Code =========================
    
     
    // =================== CPU Code =========================
    // Write our file back to disk
    const char* output = "sky-sepia.ppm";
    savePPMImage(output, myImage);

    // Free memory that we have allocated on the CPU.
    free(myImage);
    // =================== CPU Code =========================
    

    return 0;
}
