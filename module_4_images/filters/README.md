# Fun with Filters
## Introduction
In this assignment, we'll learn some different ways to manipulate images. Specifically, we will implement features such as: grayscale, increasing brightness, and filters.

## Portable Pixmap Format (PPM) 
When interacting with our images, we are going to be using the PPM format. Some useful code has been provided for you in the `common` directory to load and save these types of images. You can learn more about what each value in the file represents [here](https://en.wikipedia.org/wiki/Netpbm). Use `display <filename>` to view your images.

## RGB, Row-Major Revisited, and Parallelization Idea
In images, every pixel is comprised of 3 color components--Red, Green, and Blue. Each 8-bit value ranges from `[0,255]`, and different combinations of R,G,B make different colors.

When working with images, we now work in two dimensions (similar to matrix addition in Module 2). Each thread that we spawn is responsible for performing operations on a certain pixel `(x,y)` in our image.

Recall that in **C** our memory is stored in row-major order. So, we need to make sure we operate in "chunks" of 3, since we will operate on the R,G,B values for each pixel.

> Assume an `(x,y)` coordinate corresponds to a specific pixel, where `x` is the row, `y` is the column, `w` is the width, and `h` is the height of our image. Write an expression to get the index of the pixel (assuming row-major order) in an array.\
**TODO: Your answer here**

> Assume our image is loaded into a variable called `pixel_array`. Write an expression to get the `r`, `g`, and `b` value for an arbitrary pixel `p`.\
`r` =  **TODO: Your answer here**\
`g` =  **TODO: Your answer here** \
`b` =  **TODO: Your answer here** 

## Hints
* Introduce "clipping" for your pixel values. Introduce a maximum threshold so values can't exceed the upper limit for a valid value.
* Look the the `ppm.h` file for helpful functions to interact with PPM images.
* The questions you answered above will help you!

Some starter code has been provided. For each different filter operation below, write a new kernel call. In each task, load `safari.ppm` as your image. Save each altered image into this directory with the given filename. 

To keep things simple, launch `P` blocks, each with a single thread (where `P` is the total number values you need to keep track of for the image).

```c
/* 
 * Compile: nvcc filter.cu ./../../common/src/ppm.cu -I./../../common/include -o filter
 */

#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"

/* Converts the image to grayscale */
__global__ void make_grayscale(unsigned char* pixel_array, int width, int height) {
    /* The (x,y) coordinate within the image */
    int x = /*...TODO...*/
    int y = /*...TODO...*/

    /* Bounds check to ensure threads are within image dimensions */
    if (x < width && y < height) {
        int offset = /*...TODO...*/

        /* Extract pixel values, perform operations, modify pixel_array */
        unsigned char r = /*...TODO...*/
        ...
    }
}

int main() {
    const char* input = "safari.ppm";
    /* Load our PPM Image */
    ppm_t* myImage = /*...TODO...*/
    ...

    int numberOfPixels = /*...TODO...*/
    int sizeOfDataInBytes = (numberOfPixels)
                            * (/*...TODO...*/)
                            * (sizeof(unsigned char));

    /* Allocate required memory */ 
    ...

    cudaMalloc(/*...TODO...*/);
    cudaMemcpy(/*...TODO...*/);

    dim3 blocksPerGrid(/*...TODO...*/, /*...TODO...*/, 1);
    dim3 threadsPerBlock(/*...TODO...*/);

    make_grayscale<<< /*...TODO...*/,/*...TODO...*/ >>>(/*...TODO...*/);

    cudaMemcpy(/*...TODO...*/);
    cudaFree(/*...TODO...*/);

    const char* output = "safari-output.ppm";
    /* Save our PPM Image */
    ...
    /* Free any remaining memory on the host */
    ...
}
```
## Task 1: No filter
Get `filter.cu` to cleanly compile. Save your image as `safari_base.ppm`.

## Task 2: Grayscale
Create a kernel `make_grayscale`. Use the following formula when finding values for your pixels:
```c
R = .21f*r + .71f*g + .07f*b
G = .21f*r + .71f*g + .07f*b
B = .21f*r + .71f*g + .07f*b
```
Save your altered image as `safari_grayscale.ppm`.

## Task 3: Varying Brightness
Create two kernels: `increase_brightness` and `decrease_brightness`. 
> What types of pixel values make a color brighter? What makes it darker?\
**TODO: Your answer here**

Save your altered image as `safari_brighter.ppm` and `safari_darker.ppm`, respectively.

## Task 4: Blue Filter
Create a kernel `make_blue`. Use the following formula when finding values for your pixels:
```c
R = .189f*r + .2f*g + .769f*b
G = .49f * R
B = .69f * R
```
Save your altered image as `safari_blue.ppm`.

## Tasks
* Finish implementing `filter.cu`.
    * Complete the accompanying kernel calls and push the following images to the directory for each task.
        * `safari_base.ppm`
        * `safari_grayscale.ppm`
        * `safari_brighter.ppm`, `safari_darker.ppm`
        * `safari_blue.ppm`
    
