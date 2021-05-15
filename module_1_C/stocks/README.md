# Stock Portfolio

## Introduction
It's nearing April 15th, and after a year of investing it's time to do your taxes. You've made quite a few risky investments in the stock market, and you need to find the total value of your portfolio. Throughout the year, you have noted down each of your trades in a file named `portfolio.txt`. This file has the following format: 

```txt
GOOG 2000 2080 1
TSLA 600 572 4
FB 255 261 4
```
Each row contains:
* Ticker-tape symbol (`GOOG`)
* Purchase price (`2000`) 
* Current price (`2080`) 
* Number of shares (`1`)

## Linked List
Your task is to create a Linked List populated with the stocks from `portfolio.txt`. 

Recall that a Linked List has the following structure:

```c
typedef struct node {
    int data;
    struct node* next;
} node_t;
```
Each node has some value associated with it, as well as a pointer to the next node in the list. In this assignment, you'll need to allocate memory using `malloc` and make sure you `free` that memory before your program terminates. Each node will be represented by a `stock` struct.

## File I/O
You will have to read from a file in order to populate your Linked List. `fscanf` can help with this:
```c
int fscanf(FILE* fp, char* format, ...)
```
`fscanf` is identical to other C functions like `scanf` and `printf`, except that the first argument is a file pointer and the second argument is the desired format string. Note that the arguments to `fscanf` **must** be pointers: `fscanf(myfile, &n)` vs. `fscanf(myfile, n)`.  

Some starter code has been provided for you below:
```c
/* 
 * stocks.c 
 *
 * Read the contents of portfolio.txt and store each 
 * entry as a Linked List of stocks.
 *
 * Compile with: gcc -o stocks stocks.c
 */

#include <stdio.h>
#include <stdlib.h>

/* TODO: Finish implementing the stock_t type */
typedef struct stock {
    char* symbol;       /* Ticker-tape symbol (i.e., GOOG) */
    ...
    ...

}stock_t;

/* TODO: Declare your function prototypes here */
/* Note that you may need to change the signatures provided */
void print_portfolio(...);
void create_portfolio(...);
void free_portfolio(...);
void total_value(...);

/* Add any other helper functions you would like */

int main() {
    ...
    return 0;
}
```

## Tasks
* The defintion for a `stock` struct in the starter code is incomplete--go ahead and fix this.
* Implement the following functions:
    * `create_portfolio`: Creates a Linked List (where each element is a `stock`)
    * `free_portfolio`: Frees your Linked List (free every node that you have allocated memory for)
    * `print_portfolio`: Prints your Linked List in the following format:
        ```txt
        GOOG  | 2000  | 2080  | 1     |
        TSLA  | 600   | 572   | 4     |
        FB    | 255   | 261   | 4     |
        ```
        * You can use the "left-justify" option of `printf` to obtain the format above `printf("%-5d", baz)`
    * `total_value`: Finds the total value of the portfolio
        

You are encouraged to use a tool like `valgrind` to make sure you don't have any memory leaks in your final implementation.