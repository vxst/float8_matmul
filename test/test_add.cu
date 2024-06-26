#include <cstdio>
#include <algorithm>

#include "../matmulf8_kernel.cuh"
#include "subeq.cuh"

int a[10], b[10], c[10];
int acore[4096];

int main(){
    FILE* add_ref = fopen("test_add.bin", "rb");
    fread(a, sizeof(int), 10, add_ref);
    fread(b, sizeof(int), 10, add_ref);
    fread(c, sizeof(int), 10, add_ref);
    fclose(add_ref);

    for(int i = 0; i < 10; i++){
        // printf("%x + %x = %x\n", a[i], b[i], c[i]);
        int c_g = addv4(a[i], b[i]);
        // printf("GPU: %x\n", c);
        if(!subeq(c_g, c[i])){
            printf("%x + %x expect %x, got %x\n", a[i], b[i], c[i], c_g);
        }else{
            printf(".");
        }
    }
    printf("\n");
    return 0;
}