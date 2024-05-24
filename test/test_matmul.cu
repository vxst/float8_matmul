#include "../load_core.cuh"
#include "../matmulf8.cuh"
#include <cstdio>
#include <random>
#include "test_matmul.h"

void test_cross_line(u_int8_t *A, u_int8_t *B, u_int8_t *C, int n, int m, int p,
                     int target_x, int target_y,
                     int *acore, int *mcore){
    int success = 1;
    for(int i = 0; i < n * m; i++) {
        A[i] = 0;
    }
    for(int i = 0; i < m * p; i++) {
        B[i] = 0;
    }
    for (int i = 0; i < m; i++) {
        A[target_x * m + i] = 0x3c;
    }
    for (int i = 0; i < m; i++) {
        // B is transposed
        B[target_y * m + i] = 0x3c;
    }
    for (int i = 0; i < n * p; i++) {
        C[i] = 0xff;
    }

    float t = matmul((int *)A, (int *)B, (int *)C, n, m, p, acore, mcore);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            if (i == target_x && j == target_y){
                // 0x58 is float8_e5m2 of 128.0
                if(C[i * p + j] != 0x58){
                    printf("Target(%d, %d) expected 0x58(128.0), got: %x\n", i, j, C[i * p + j]);
                    success = 0;
                }
            }else{
                if(C[i * p + j] != 0){
                    printf("(%d, %d) expected 0, got: %x", i, j, C[i * p + j]);
                    success = 0;
                }
            }
        }
    }
    if(!success){
        printf("Test failed for (%d, %d)\n", target_x, target_y);
    }else{
        printf("Test passed for (%d, %d)\n", target_x, target_y);
    }
}

int main(){
    cudaSetDevice(0);

    int* acore = load_core("../cores/f8e5m2_acore.bin");
    int* mcore = load_core("../cores/f8e5m2_mcore.bin");
    int n = 128, m = 128, p = 128;
    u_int8_t *A, *B, *C;

    cudaMallocHost(&A, n * m);
    cudaMallocHost(&B, m * p);
    cudaMallocHost(&C, n * p);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dis(0, 128);

    for(int t = 0; t < 10; t++){
        int x = dis(gen), y = dis(gen);
        test_cross_line(A, B, C, n, m, p, x, y, acore, mcore);
    }

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
}
