#include <cstdio>

#include <cuda_runtime.h>
#include "load_core.cuh"
#include "matmulf8_kernel.cuh"

float matmul(int* A, int* B, int* C, int n, int m, int p, int* acore, int* mcore) {
    int* d_A, *d_B, *d_C, *d_acore, *d_mcore;
    cudaMalloc(&d_A, n * m / 4 * sizeof(int));
    cudaMalloc(&d_B, m * p / 4 * sizeof(int));
    cudaMalloc(&d_C, n * p / 4 * sizeof(int));
    cudaMalloc(&d_acore, 16384);
    cudaMalloc(&d_mcore, 16384);
    cudaMemcpy(d_A, A, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * p * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acore, acore, 16384, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mcore, mcore, 16384, cudaMemcpyHostToDevice);

    float t;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Each kernel thread do 4xf8 result
    matmulf8<<<dim3(n / 32, p / 32), dim3(32, 8)>>>(d_A, d_B, d_C, n, m, p, d_acore, d_mcore);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);

    cudaMemcpy(C, d_C, n * p * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_acore);
    cudaFree(d_mcore);

    return t;
}

int main() {
    int n = 1024, m = 1024, p = 1024;
    int *A, *B, *C;
    cudaMallocHost(&A, n * m * sizeof(int) / 4);
    cudaMallocHost(&B, m * p * sizeof(int) / 4);
    cudaMallocHost(&C, n * p * sizeof(int) / 4);
    int* acore = load_core("addcore.bin");
    int* mcore = load_core("mltcore.bin");
    for(int i = 0; i < n * m; i++) {
        A[i] = rand() & 0xff;
    }
    for(int i = 0; i < m * p; i++) {
        B[i] = rand() & 0xff;
    }
    float t = matmul(A, B, C, n, m, p, acore, mcore);
    printf("Time: %f ms\n", t);
    cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C);
    cudaFreeHost(acore); cudaFreeHost(mcore);
    return 0;
}