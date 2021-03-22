# Kernel Calls

## Introduction
As shown in lecture, kernel code lets our compiler know that certain code should be run on the device (our GPU) instead of our host macine. We use `<<<NUM_BLOCKS, NUM_THREADS>>>` to specify our memory arguments, and to tell CUDA how many threads and blocks we would like to launch. Threads are grouped into **blocks** called "**thread blocks**," and multiple **thread blocks** make up a "**grid**." Each thread and block maps to specific type of processor on the GPU. 

## Addition
Given below is a CPU implementation for a function `add` that adds two numbers. Your task is to create a CUDA program (`add.cu`) that performs the same operation, except this time makes a kernel call. You need to ensure that you allocate memory on the device, transfer your data over, and free the used memory once you are done. Use 1 block and 1 thread when performing this operation.

```c
#include <stdio.h>

/*
 * add.c
 * 
 * Adds two numbers using the CPU.
 */

int add(int a, int b, int* c) {
    *c = a + b;
}

int main() {
    int c;
    multiply(6, 7, &c);
    printf("CPU says: 3 + 9 = %d", c);
}

```

## Multiplication
Similar to what we did above, write a C program (`multiply.c`) that contains a function `multiply` that multiplies two numbers. Then, write a CUDA program (`multiply.cu`) that performs the same operation. Use 1 block and 1 thread when crafting your call to the kernel.

## Tasks
* Implement `add.cu` 
* Implement `multiply.c` and `multiply.cu`
