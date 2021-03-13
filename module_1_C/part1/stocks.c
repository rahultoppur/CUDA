#include <stdio.h>
#include <stdlib.h>

/* 
 * stocks.c 
 *
 * Read the contents of portfolio.txt and store each 
 * entry as a Linked List of stocks.
 *
 * Compile with: gcc -o stocks stocks.c
 */


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

    return 0;

}
