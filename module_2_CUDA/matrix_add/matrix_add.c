#include <stdio.h>
#include <stdlib.h>

#define N 37000


void matrixAdd(int* a, int* b, int* c) {
    for(int i=0; i < N; ++i) {
        for(int j=0; j < N; ++j) {
            c[i*N + j] = a[i*N + j] + b[i+N + j];
        }
    }
}

void printMatrix(int* m) {
    for(int i=0; i < N; ++i) {
        printf("[");
        for(int j=0; j < N; ++j) {
            printf("%-5d", m[i*N +j]);
        }
        printf("]");
        printf("\n");
    }
    printf("\n");
}

int main() {
    //int a[N][N], b[N][N], c[N][N];
    int* a = malloc(N * N * sizeof(int));
    int* b = malloc(N * N * sizeof(int));
    int* c = malloc(N * N * sizeof(int));


    /* Populate elements in a and b */
    for(int i=0; i < N; i++) {
        for(int j=0; j < N; j++) {
            a[i*N + j] = i + j;
            b[i*N + j] = i - j;
        }
    }
    
    matrixAdd(a, b, c);

//    printMatrix(a);
//    printMatrix(b);
//    printMatrix(c);

    free(a);
    free(b);
    free(c);


    /* Display our matrices */
    

    /* Populate elements in b */
}

