// Copyright (C) 2024 Chunqing Shan
// 
// float8_matmul is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// float8_matmul is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with float8_matmul. If not, see <http://www.gnu.org/licenses/>.

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
    cudaMemcpy(d_A, A, n * m * sizeof(int) / 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * p * sizeof(int) / 4, cudaMemcpyHostToDevice);
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

    cudaMemcpy(C, d_C, n * p * sizeof(int) / 4, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_acore);
    cudaFree(d_mcore);

    return t;
}

int main() {
    int n = 4096, m = 4096, p = 4096;
    int *A, *B, *C;
    cudaSetDevice(0);
    cudaFree(0);
    
    cudaMallocHost(&A, n * m * sizeof(int) / 4);
    cudaMallocHost(&B, m * p * sizeof(int) / 4);
    cudaMallocHost(&C, n * p * sizeof(int) / 4);
#ifdef DB
    int* acore = load_core("addcore.bin");
#else
    int* acore = load_core("apdcore.bin");
#endif
    int* mcore = load_core("mltcore.bin");
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