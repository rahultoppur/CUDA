#include <stdio.h>

/* 
   multiply.cu

   Multiplies two numbers using the GPU.

*/



__global__ void multiply(int a, int b, int* c) {
    *c = a * b;
}

int main() {
    int c;
    int* dev_c;

    cudaMalloc(&dev_c, sizeof(int));

    multiply<<<1,1>>>(3, 9, dev_c);
    
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU says that: 3 * 9 = %d\n", c);
    
    cudaFree(dev_c);

    return 0;
}
