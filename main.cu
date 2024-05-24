#include <cuda_runtime.h>
#include <cstdio>

#include "matmulf8.cuh"
#include "load_core.cuh"


int main() {
    int n = 4096, m = 4096, p = 4096;
    int *A, *B, *C;
    cudaSetDevice(0);
    cudaFree(0);
    
    cudaMallocHost(&A, n * m * sizeof(int) / 4);
    cudaMallocHost(&B, m * p * sizeof(int) / 4);
    cudaMallocHost(&C, n * p * sizeof(int) / 4);
#ifdef DB
    int* acore = load_core("cores/f8e5m2_adbcore.bin");
#else
    int* acore = load_core("cores/f8e5m2_acore.bin");
#endif
    int* mcore = load_core("cores/f8e5m2_mcore.bin");
    for(int i = 0; i < n * m / 4; i++) {
        A[i] = rand();
        // A[i] &= 0x7f7f7f7f;
        A[i] = 0;
    }
    for(int i = 0; i < m * p / 4; i++) {
        B[i] = rand();
        // B[i] &= 0x7f7f7f7f;
        B[i] = 0;
    }
    float t = matmul(A, B, C, n, m, p, acore, mcore);
    printf("Time: %f ms\n", t);
    float flops = 2.0 * n * m * p / t / 1e6;
    printf("FLOPS: %f GFLOPS\n", flops);
    cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C);
    cudaFreeHost(acore); cudaFreeHost(mcore);
    return 0;
}