#ifndef PPM_H
#define PPM_H


// The structure represents an image.
typedef struct{
    // The magic number tells us what type of PPM
    // image we are working with, that is, how is
    // the data formatted within the file.
    int magicnumber;
    // The width of our image
    int width;
    // The height of our image
    int height;
    // The range of values for a particular
    // component of a pixel.
    // For example, 0-255 is often used to tell
    // the intensity of the Red, green, and blue
    // color components of a pixel
    int range;
    // This array stores the raw values of the pixels
    unsigned char* pixels;
}ppm_t;

// Some helper functions found in "./src/ppm.cu"
// We make these functions 'extern' because they are in an external module.
// Note that we need to compile the ppm.cu file
// which has the implementations of these functions.
// Anywhere we otherwise include 'ppm.h' we are able
// to use both of these functions.

extern ppm_t* loadPPMImage(const char* filepath);
extern void savePPMImage(const char* filename, ppm_t* ppm);

#endif
