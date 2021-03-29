#include <stdio.h>
#include <stdlib.h>
#define N 25000
#define BLOCK_DIM 32


__global__ void matrixAdd(float* a, float* b, float* c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * N;

    if (col < N && row < N) {
        c[index] = a[index] + b[index];
    }
}

/*
void printMatrix(long* m) {
    for(long i=0; i < N; ++i) {
        printf("[");
        for(long j=0; j < N; ++j) {
            printf("%-5d", m[N*i + j]);
        }
        printf("]");
        printf("\n");
    }
    printf("\n");
}
*/

int main() {
    //long a[N][N];
    //long b[N][N];
    //long c[N][N];

    size_t size = N * N * sizeof(float);

    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);
    
    float* dev_a;
    float* dev_b; 
    float* dev_c;

    /* Populate elements long* a and b */
    for(int i=0; i < N; i++) {
        for(int j=0; j < N; j++) {
            a[N*i + j] = i + j;
            b[N*i + j] = i - j;
        }
    }

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    //dim3 dimBlock(22, 22);
    dim3 dimGrid((int)ceil((double)N/dimBlock.x), (int)ceil((double)N/dimBlock.y));

    matrixAdd<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    

    //prlongMatrix(a);
    //prlongMatrix(b);
    //prlongMatrix(c);

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

}

