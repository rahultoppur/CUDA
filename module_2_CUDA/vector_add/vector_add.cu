#include <stdio.h>
#include <time.h>

#define N 1000000

__global__ void add(int* a, int* b, int* c) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id < N) {
        c[t_id] = a[t_id] + b[t_id];
    }
}

int main() {
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    */

    clock_t start, end;

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
    for(int i=0; i<N; ++i) {
        a[i] = -1 * i;
        b[i] = 2 * i;
    }

    // copy vectors a and b to the GPU
    // at this point, we have a valid address in dev_a
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // perform the vector add with the kernel
    //cudaEventRecord(start);
    start = clock();
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)N/blockSize);
    add<<<gridSize,blockSize>>>(dev_a, dev_b, dev_c);
    end = clock();
    //cudaEventRecord(stop);

    // copy answer back from device to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // display the results
    //for(int i=0; i<N; ++i) {
    //    printf("%-4d + %-4d = %-4d\n", a[i], b[i], c[i]);
    //}

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
    printf("Elapsed Time: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

}
