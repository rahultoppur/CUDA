# The Caesar Cipher Revisited
## Introduction
In Part 2 of Module 1, we cracked some ciphertext encrypted using the Caesar Cipher by brute-forcing all possible combinations of the key, **k**. We are now going to revisit that same idea, but this time we will parallelize our implementation using CUDA.

## Parallelization Idea
Previously, we had to have two nested `for` loops in `caesar.c`. The outer for loop tried all possible values of the key while the inner for loop iterated through each character in our ciphertext. This time, we'll launch N separate threads--each thread will be responsible for trying a value of k. As soon as each thread has finished, we write its output to a buffer, and continue. 

> What should the value of `N` be? (e.g., how many threads should we launch?)\
**TODO: Your answer here**

Some psuedocode has been provided for you below:
```c
/*
 * caesar.cu 
 * 
 * Decrypts a message encrypted with the Caesar cipher.
 */

#include <stdlib.h>
#include <string.h>

/* Number of threads to spawn */
#define N /*...TODO ...*/

__global__ void caesar_decrypt(char* msg, char* msg_output) {
    /* Include a bounds check here so we don't access 
       elements outside of N */

    /* Each thread that we spawn tries a value of k */
    ...
    /* Logic for decrypting Caesar Shift */
    ...
    /* As soon as each thread finishes its work, it
       writes to a unique section of msg_output */
    ...
}

int main() {
    char msg[] = "sncr wi rnm cwzihmmd jmzyuwrk yv cp apxcdmp iscxxys";

    /* Allocate memory on the GPU, copy it over,
       and call your kernel. */
    ...
    /* Display the contents of msg_output */
    ...
    /* Don't forget to free host and device memory
       when done! */
    ...
}
```
## Tasks
* Implement `caesar.cu`, defining a constant N which represents the number of threads to spawn in your program.
* Finish implementing the `caesar_decrypt` kernel call. Spawn a single block of N threads.
* Output each individual try for **k** to the console. For example:
    * ```txt
        dahhkTpdana
        czggjSoczmz
        byffiRnbyly
        ...
      ```
* >What did the message decrypt to?\
**TODO: Your answer here**