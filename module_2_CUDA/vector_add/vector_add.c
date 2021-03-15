#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Sum two vectors.
 */

#define N 1000000

void vector_add(int* a, int* b, int* c) {
    for(int i=0; i<N; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;

    //int a[N], b[N], c[N];
    int* a = (int*)malloc(sizeof(int) * N);
    int* b = (int*)malloc(sizeof(int) * N);
    int* c = (int*)malloc(sizeof(int) * N);
    
    // Populate our arrays with data
    for (int i=0; i < N; ++i) {
        a[i] = -1 * i;
        b[i] = 2 * i;
    } 

    start = clock();
    vector_add(a, b, c);
    end = clock();

    //for(int i=0; i<N; ++i) {
    //    printf("%-4d + %-4d = %-4d\n", a[i], b[i], c[i]);
    //}


    free(a);
    free(b);
    free(c);

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %f\n", time_spent);
}
