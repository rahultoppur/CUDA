#include <stdio.h>

/* 
 * multiply_cpu.c
 *
 * Multiply two numbers using the CPU.
 */

int multiply(int a, int b, int* c) {
    *c = a * b;
}

int main() {
    int c;
    multiply(3, 9, &c);  
    printf("CPU says: 3 * 9 = %d\n", c);
}
