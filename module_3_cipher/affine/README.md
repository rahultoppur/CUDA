# The Affine Cipher
## Introduction
As we wrap up our study of ciphers and CUDA, we will decrypt a message encrypted with the Affine cipher. Like the Caesar-shift, the Affine is also a monoalphabetic substitution cipher. However, each alphabet is now encrypted and decrypted using the respective mathematical function:

<div style="text-align:center"><img src="../../media/affine.png" width="400" height="92"></div>

## Parallelization Idea
We first need to identify how multiple threads can help us solve this problem. Ideally, since we have our decryption function `D(x)`, we should be able to operate on each letter individually. 

> How many threads should we spawn to decrypt a message of length `m`? How is this different from our previous approach with the Caesar cipher?\
**TODO: Your answer here**

You tasks is to implement `affine.cu`. Name your kernel `affine_decrypt`, and use the following values when writing the logic for `D(x)`.
* `A` = `5`
* `B` = `9`
* `A_MMI_M` = `21` (Modular Multiplicative Inverse)
* `M` = `?`
> What should the value of M be? Look to see the types of characters in `msg`.\
**TODO: Your answer here**

Define these as constants in your program. Spawn a single block with the appropriate number of threads.

Some starter code for `affine.cu` has been provided for you.
```c
#include <stdio.h>
#include <stdlib.h>

#define N /* ... TODO ... */

/* modulo function that works with negative numbers */
int modulo(int a, int b) {
    int r = a % b;
    if (r < 0) { return r + b; }
    return r;
}

__global__ void affine_decrypt(char* msg, char* msg_output) {
    /* Include bounds check here */
        ...
    /* Work on a separate portion of msg each time */
        ...
    /* Logic for decrypting Affine */
        ...
    /* As each thread finishes its work, it
       writes to a unique section of msg_output */
        ...
}

__global__ void affine_decrypt_block(char* msg, char* msg_output) {
    /* ... TODO ... */
}

int main() {
    char msg[] = "orizwuzrciwpufccxjclaswzyahivevlixcvuoillao";
    /* Allocate memory on the GPU, copy it over,
       and call your kernel. *
        ...
    /* Display the contents of msg_output*/
        ...
    /* Free host and device memory when done */
        ...
}
```
## Varying the Number of Blocks
Define another kernel, this time called `affine_decrypt_block`. Note that you now need to change the way you find each thread's ID. Once implemented, spawn `N` blocks, each with a single thread.

## Hints
As a first step, you should obtain the position in the normal alphabet of each letter in `msg`. This value is represented by `x` in `D(x)`. `char` to `int` conversions would be helpful for this purpose.

## Tasks
* Finish implementing `affine.cu`:
    * Implement the `affine_decrypt` kernel call
    * Implement the `affine_decrypt_block1` kernel call
* Output each individual try for **k** to the console. For example:
    * ```txt
        dahhkTpdana
        czggjSoczmz
        byffiRnbyly
        ...
      ```
* >What did the message decrypt to?\
**TODO: Your answer here**









