#include <stdio.h>
#include <time.h>

#define N 10000
#define T 256

__global__ void add(int* a, int* b, int* c) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    //int stride = blockDim.x * gridDim.x;
    //for (int i=t_id; i < N; i+= stride){
    //    c[i] = a[i] + b[i];
    //}

    if (t_id < N) {
        c[t_id] = a[t_id] + b[t_id];
    }
    

    //int i = threadIdx.x;
    //printf("i is: %d\n", i);
    //c[i] = a[i] + b[i];    
    
    //int index = threadIdx.x;
    //int stride = blockDim.x;
    //int stride = blockDim.x;

    //for(int i=index; i<N; i += stride){
    //    printf("index is: %d\n", index);
    //    c[i] = a[i] + b[i];
    //}
}

int main() {
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    */


    /*
    int a[N];
    int b[N];
    int c[N];
    */
    int* a = (int*)malloc(sizeof(int) * N);
    int* b = (int*)malloc(sizeof(int) * N);
    int* c = (int*)malloc(sizeof(int) * N);


    int* dev_a; // "reserve" the seat
    int* dev_b;
    int* dev_c;

    /*
    printf("dev_a: %x\n", &dev_a);
    printf("dev_b: %x\n", &dev_b);
    printf("dev_c: %x\n", &dev_c);
    */

    // regular malloc gives us a pointer to our memory. cudaMalloc however
    // makes us pass in a handle ourselves. Then, we still have a "handle"
    // or access, to memory for our GPU.

    // allocate our memory on the GPU
    cudaMalloc(&dev_a, N * sizeof(int)); // get back the "group of seats" that you've reserved
    cudaMalloc(&dev_b, N * sizeof(int));
    cudaMalloc(&dev_c, N * sizeof(int));

    // fill vectors a and b with data
    for(int i=1; i<N; ++i) {
        a[i] = -1 * i;
        b[i] = 2 * i;
    }

    // copy vectors a and b to the GPU
    // at this point, we have a valid address in dev_a
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // perform the vector add with the kernel
    add<<<1,T>>>(dev_a, dev_b, dev_c); // Review this in more detail...
    //add<<<(int)ceil(N/T),T>>>(dev_a, dev_b, dev_c); // Review this in more detail...

    // copy answer back from device to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // display the results
    //for(int i=0; i<N; ++i) {
    //    printf("%-4d + %-4d = %-4d\n", a[i], b[i], c[i]);
    //}

    printf("Passed!\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
/*
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    */

}
