#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Things students learn: 
 * File IO
 * strings, pointers, malloc, free
 * Ask them to:
 * implement few more stubs (like free method)
 * find value of portfolio
 * iterator pattern (useful)
 * check mem leaks with valgrind
 */


// Representation for each stock
typedef struct stock {
    char* symbol;   // Ticket-tape symbol (i.e., GOOG)
    int buyPrice;       // Price we bought the stock for
    int currPrice;      // Current price of the stock
    int numShares;      // Number of shares that we own
    struct stock* next; // The next stock in our portfolio

}stock_t;

// Declare function prototypes here...
void printStock(stock_t* stock);

stock_t* createStock(char* symbol, int buyPrice, int currPrice, int numShares, stock_t* next_stock) {
    // First, create our new node
    stock_t* newStock = NULL;
    newStock = (stock_t*)malloc(sizeof(stock_t)); 
    newStock->symbol = (void*)malloc(strlen(symbol)+1);
    strcpy(newStock->symbol, symbol);
    newStock->buyPrice = buyPrice;
    newStock->currPrice = currPrice;
    newStock->numShares = numShares;
    newStock->next = next_stock;
    printStock(newStock);
    return newStock;
}

void append(char* symbol, int buyPrice, int currPrice, int numShares, stock_t* head) {
    stock_t* tmp = head;
    while(tmp->next != NULL) {
        tmp = tmp->next;
    }
    stock_t* stock_to_append = createStock(symbol, buyPrice, currPrice, numShares, NULL);
    tmp->next = stock_to_append;
}

stock_t* create_list(char* filename) {

    FILE* portfolio = fopen(filename, "r");
    int buyPrice, currPrice, numShares;
    char symbol[10];


    if (portfolio == NULL) {
        fprintf(stderr, "%s.txt not found...make sure it's in the current directory!", filename);
        exit(1);
    }
    // We have successfully opened our file
    
    printf("opened file\n");

    // Create a "dummy" head node to start populating our data 
    stock_t* headNode;
    printf("created head node\n");
    headNode = createStock("foo",-1,-1,-1,NULL);


    while(fscanf(portfolio, "%s %d %d %d", symbol, &buyPrice, &currPrice, &numShares) != EOF) {
        //printf("%s %d %d %d\n", &symbol, buyPrice, currPrice, numShares);
        append(symbol, buyPrice, currPrice, numShares, headNode); 
        // Want to append our data here to our linked list
    }

    return headNode;
}

void printStock(stock_t* tmp) {
    //stock_t* tmp = head;
    //while(tmp != NULL) {
    printf("%-5s | %-5d | %-5d | %-5d |\n", tmp->symbol, tmp->buyPrice, tmp->currPrice, tmp->numShares);
    //}
    //tmp = tmp->next;

    //while(head != NULL) {
     //   head = head->next;
    //}
}



// Read in our portfolio file
int main() {
    stock_t* myStocks = create_list("./portfolio.txt");
    printStock(myStocks);


}
