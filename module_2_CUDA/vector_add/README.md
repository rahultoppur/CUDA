# Vector Addition

## Introduction
Vectors have an important role in the world of physics and computer graphics. In this section (and later on when we deal with matrices), we'll take a deeper look at the **memory arguments** of a kernel call and observe the affect varying the number of blocks and threads has on the runtime of our programs.


## C Implementation (`vector_add.c`) and a Quick Refresher
Before you start implementing `vector_add.c`, briefly recall the thought process behind vector addition for vectors of an arbitrary length `N`. For now, we represent each vector as an integer array.

<img src="../../media/vector_add.png" width="300" height="100">

Provided is some starter code for the **C** implementation of vector addition (`vector_add.c`). 

```c
#include <stdio.h>
#include <stdlib.h>

/*
 * vector_add.c A
 * 
 * Adds two vectors and displays the result.
 */

/* Number of elements in each vector */
 #define N 1000000

 void vector_add(int* a, int* b, int* c) {
     /* TODO */
     ...
 }

 int main() {
     /* Allocate space for vectors a, b, and c */
     ...
     /* Populate vectors with values */
     ...
     /* Display the answer (print vector c to stdout) */
     ...
     /* Free memory when done */
 }
```
## CUDA Implementation

Create the corresonding CUDA implementation, but leave your memory arguments `<<<NUM_BLOCKS,NUM_THREADS>>>(dev_a, dev_b, dev_c)` when making the `vector_add` kernel call **empty**. We will vary the number of blocks and threads and gather some performance data using `nvprof` in the sections below. 

## Scenario 1: Establishing a Baseline (`vector_add_baseline.cu`)
To establish a "baseline" to judge our performance, spawn 1 block with 1 thread when making your call to the kernel. You shouldn't have to change the implementation of your `vector_add` function. Set `N` = `1000000`.

Using the UNIX command `time`, measure the `real` time it takes for `vector_add.c` to run. For example, if the executable for `vector_add.c` is `vector_add_cpu`, you can run: `time ./vector_add_cpu`. Repeat this process once more, but this time give your CUDA exectuable.
  
After obtaining both of these values, put them in the table below:
| Implementation | Time (seconds) | 
| :--- | :--- |
|`C` (`vector_add.c`) | **TODO: Your answer here** |
|`CUDA` (`vector_add_baseline.cu`) | **TODO: Your answer here** |

What do the times look like for each program? Is one implementation slower than the other? If so, explain the reason for the resulting discrepancy.\
**TODO: Your answer here**

Using [`nvprof`](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview) (a tool that lets you look at the metrics for CUDA kernels), find the `Avg` amount of time spent within the kernel. What CUDA `API` call was called the most?\
**TODO: Your answer here**

## Scenario 2: Single Block, Many Threads (`vector_add_threads.cu`)
Now that we have established our baseline measurement with `N` = `1000000` elements, let's see how we can speed things up. We'll make use of SIMT (Single Instruction, Multiple Threads) to achieve high parallel performance for our vector computation. This time, create `vector_add_threads.cu` where you spawn `1` block with `256` threads. Define a constant `T` to represent the number of threads. The idea here is that in parallel, each thread will be responsible for performing a step of the vector addition.

Note that you will now have to make some changes to your `vector_add` kernel. If we only changed the number of threads in our memory arguments, we would be repeating the same computation once per thread instead of spreading it across multiple, parallel threads.

For example, we want Thread ID `0` to be responsible for adding the first element of both vectors. Once it has done its job, it should start solving a new portion of the problem--ideally, the 256th element of both vectors. 

Modify your existing kernel's loop to account for this behavior (ideally, spawning a new thread every iteration that solves a different portion of the problem). `blockIdx.x` and `threadIdx.x` might be helpful here.

Refer to the pseudocode below:
```c
__global__ void vector_add(int* a, int* b, int* c) {
    /* Find the unqiue ThreadID of each thread that we spawn */
    ...
    /* Find the value we need to increment by to work on a new
       portion of the problem each time */
    ...
    /* Logic to add vectors together */
    ...
}
```
Use `nvprof` to run your implementation on a vector with `N` = `1000000` elements. Note down the time spent in the kernel.

## Scenario 3: Introducing Blocks (`vector_add_blocks.cu`)
Finally, we are going to see the impact on performance when adding multiple blocks and threads. The idea is to split up the work of addition between different blocks, each with `N` threads. Each thread will process a single row operation for the vector addition. Once again, launch `256` threads. Define a constant `T` to represent the number of threads.

When determining the number of blocks you need, note that each block is reserved completely. Functions like `ceil` and `(int)` type-casting might be helpful here.

Write the memory arguments that you would pass in to your kernel call. Your answer should be in terms of `N` (number of elements in your vector) and `T` (number of threads we are spawning).\
`<<<**TODO: Your answer here**>>>`

In addition to changing your memory arguments, you will need to re-calculate the Thread ID within your kernel call (`vector_add`). You now need to compute the **Global** Thread ID with respect to the block you are in as well.

Write the snippet for calculating the Global Thread ID. Feel free to make use of CUDA's built-in variables, such as `threadIdx.x`, etc.\
`Global Thread ID = **TODO: Your answer here**`

Finally, modify the kernel by adding a bounds check so writes beyond the bounds of allocated memory are not allowed. Ensure that threads with a given Thread ID do not exceed `N`.

Use `nvprof` to run your implementation on a vector with `N` = `1000000` elements. Note down the time spent in the kernel.

## Analysis
Fill in the table with your results from Scenarios 1-3:
| Scenario | Time (ms) | Speedup |
| :--- | :--- | :--- |
|1: Baseline | **TODO** | 1.00x|
|2: Threads | **TODO** | **TODO** |
|3: Blocks and Threads | **TODO** | **TODO**|

What do your observations show? What effect do threads and blocks have on the performance of vector addition?

## Tasks
* Implement `vector_add.c`
* Complete the **short answer questions** and **tables** in Scenarios 1-3, and implement the following files:
    * `vector_add_baseline.cu`
    * `vector_add_threads.cu`
    * `vector_add_blocks.cu` 

