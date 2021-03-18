# Vector Addition

## Introduction
Vectors have a very important role in the world of physics and computer graphics. In this section (and later on when we deal with matrices), we'll take a deeper look at the **memory arguments** of a kernel call and observe the affect varying the number of blocks and threads has on the runtime of our programs.

Provided is some starter code for the **C** implementation of vector addition (`vector_add.c`). Create the corresonding CUDA implementation, but leave your memory arguments `<<<NUM_BLOCKS,NUM_THREADS>>>(dev_a, dev_b, dev_c)` empty. We will vary these values in each section below and gather some data using `nvprof`. 

Before you start implementing `vector_add.c`, briefly recall the thought process behind vector addition for vectors of an arbitrary length.

![Vector Addition](../../media/vector_add.png)

## 1 Block, 1 Thread


