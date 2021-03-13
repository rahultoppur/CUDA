/* 
 * Objective: Decrypt caesar cipher using brute-force method. 
 * Read message from a file. Try to decrypt it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main() {
    // change to hello there, general kenobi
	//char msg[] = "hello there";
	char msg[] = "dahhk pdana";

    for (int j=0; j < 26; ++j) {
        for (int i=0; msg[i] != 0; ++i) {
            //printf("%c", (msg[i] - 'a' + 3) % 26 + 'a');
            printf("%c", (msg[i] - 'a' - j + 26) % 26 + 'a');
            //printf("%c\n", msg[i]);
        }
        printf("\n");
    }

}
