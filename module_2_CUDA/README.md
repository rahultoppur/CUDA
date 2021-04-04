# CUDA
## Introduction
In this assignment, we will get more experience writing programs that utilize our CUDA-enabled GPU. Additionally, we'll perform a small case study that looks at the resulting performance of GPUs vs. CPUs for certain types of programs. 

This assignment will cover:
* Calls to the kernel
* Vector Addition
* Matrix Addition

## Assignment
There are three parts to this assignment:
* [Part 1](./multiply/README.md) -- Kernel Calls
* [Part 2](./vector_add/README.md) -- Vector Addition
* [Part 3](./matrix_add/README.md) -- Matrix Addition

## Module Learning Objectives
* Introduce making calls to the kernel
* Cover `cudaMalloc` and `cudaMemcpy` operations
* Recognize similarities between vector addition and matrix addition
* Understand memory arguments
* Use `nvprof` to see how much time is spent in the kernel
* Use the `time` UNIX utility
* Introduction to threads, blocks, and resulting performance gain
* Understand the different ways we can get a thread's index
* Writing a grid-stride loop vs. monolithic kernels
* Understand row-major order (memory layout)
* Make a graph to visualize performance
* Write C code and its CUDA equivalent


## Slides

## Presentation Video

## References
* https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays