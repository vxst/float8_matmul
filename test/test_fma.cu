#include <cstdio>

#include "../matmulf8_kernel.cuh"
#include "subeq.cuh"


int a[10], b[10], c[10], d[10];
int mcore[4096];

int main(){
    FILE* fma_ref = fopen("test_fma.bin", "rb");
    fread(a, sizeof(int), 10, fma_ref);
    fread(b, sizeof(int), 10, fma_ref);
    fread(c, sizeof(int), 10, fma_ref);
    fread(d, sizeof(int), 10, fma_ref);
    fclose(fma_ref);

    FILE* mltcore = fopen("../cores/f8e5m2_mcore.bin", "rb");
    fread(mcore, sizeof(int), 4096, mltcore);
    fclose(mltcore);

    for(int i = 0; i < 10; i++){
        // printf("%x * %x + %x = %x\n", a[i], b[i], c[i], d[i]);
        int d_fma = fma8v4(a[i], b[i], c[i], mcore);
        // printf("GPU: %x\n", d_fma);
        if(!subeq(d_fma, d[i])){
            printf("%x * %x + %x expect %x, got %x\n", a[i], b[i], c[i], d[i], d_fma);
        }else{
            printf(".");
        }
    }
    printf("\n");
    return 0;
}