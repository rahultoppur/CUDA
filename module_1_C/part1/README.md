# Stock Portfolio

## Introduction
It's nearing April 15th, and after a year of investing it's time to do your taxes. You've made quite a few risky investments in the stock market, and you need to find the total value of your portfolio. Throughout the year, you have noted down each of your trades in a file, `portfolio.txt`. This file is in the following format:

```txt
GOOG 2000 2080 1
TSLA 600 572 4
FB 255 261 4
```
Each row contains:
* ticker-tape symbol (`GOOG`)
* purchase price (`2000`) 
* current price (`2080`) 
* number of shares (`1`)

## Linked List
Your task is to create a linked list populated with the stocks from `portfolio.txt`. 

Recall that a linked list takes the following structure:

```c
typedef struct node {
    int data;
    struct node* next;
} node_t;
```
Each node has some value associated with it, as well as a pointer to the next node in the list. In this assignment, you'll need to allocate memory using `malloc` and make sure you `free` that memory before your program terminates.

## File I/O
You will have to read from a file in order to populate your liked list. `fscanf` can help with this:
```c
int fscanf(FILE* fp, char* format, ...)
```
`fscanf` is identical to other C functions like `scanf` and `printf`, except that the first argument is a file pointer and the second argument is the desired format string. Note that the arguments to `fscanf` **must** be pointers: `fscanf(myfile, &n)` vs. `fscanf(myfile, n)`.  

## Tasks
* The defintion for a `stock` struct in the starter code is incomplete--go ahead and fix this.
* Implement the following functions:
    * `create_portfolio`: Creates a linked list (where each element is a `stock`)
    * `free_portfolio`: Frees your linked list (free every node that you have allocated memory for)
    * `print_portfolio`: Prints your linked list in the following format:
        ```
        GOOG  | 2000  | 2080  | 1     |
        TSLA  | 600   | 572   | 4     |
        FB    | 255   | 261   | 4     |
        ```
    * `total_value`: Finds the total value of the portfolio.
        

You are encouraged to use a tool like `valgrind` to make sure you don't have any memory leaks in your final implementation.

Some starter code has been provided for you. Feel free to create any other helper functions you need for this assignment.
