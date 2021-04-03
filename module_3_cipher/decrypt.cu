#include <stdio.h>
#include <stdlib.h>

// Make sure you are getting the right A->Z conversion for the cipher alphabet first.

/* Total number of characters in our message */
#define M 52
#define A 9
#define B 2
#define A_MMI_M 4


__global__ void affine_decrypt(int* d_input, int* d_output) {
    int index = threadIdx.x;
    int value = d_input[index];
    value = (A_MMI_M * (value - B)) % M;
    d_output[index] = value;
}


/*
__global__ void affine_decrypt_multiblock(int* d_input, int* d_output) {
    
}
*/

int main() {
    int s = M * sizeof(int);

    int* dev_ciphertext;
    int* dev_ciphertext_decrypted;

    /* Allocate host memory */
    //int* ciphertext = (int*)malloc(size);
    // we want to read this value from a file later...
    char ciphertext[] = "SNCRWIRNMCWZIHMMDJMXYUWRKYVCPAPXCDMPISCXXYS";
    //char ciphertext[] = "sncr wi rnm cwzihmmd jmzyuwrk yv cp apxcdmp iscxxys";
    int* plaintext = (int*)malloc(s);

    /* Allocate device memory */
    cudaMalloc((void**)&dev_ciphertext, s);
    cudaMalloc((void**)&dev_ciphertext_decrypted, s);

    cudaMemcpy(dev_ciphertext, ciphertext, s, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ciphertext_decrypted, plaintext, s, cudaMemcpyHostToDevice);

    affine_decrypt<<<1,M>>>(dev_ciphertext, dev_ciphertext_decrypted);

    cudaMemcpy(plaintext, dev_ciphertext_decrypted, s, cudaMemcpyDeviceToHost);

    /* Display our result */
    for(int i=0; i < M; i++) {
        printf("%d", plaintext[i]);
    }
     

}

