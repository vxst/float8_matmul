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

float matmul(int* A, int* B, int* C, int n, int m, int p, int* mcore) {
    int* d_A, *d_B, *d_C, *d_mcore;
    cudaMalloc(&d_A, n * m / 4 * sizeof(int));
    cudaMalloc(&d_B, m * p / 4 * sizeof(int));
    cudaMalloc(&d_C, n * p / 4 * sizeof(int));
    cudaMalloc(&d_mcore, 16384);
    cudaMemcpy(d_A, A, n * m * sizeof(int) / 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * p * sizeof(int) / 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mcore, mcore, 16384, cudaMemcpyHostToDevice);

    float t;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Each kernel thread do 4xf8 result
    matmulf8<<<dim3(n / 32, p / 32), dim3(32, 8)>>>(d_A, d_B, d_C, n, m, p, d_mcore);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);

    cudaMemcpy(C, d_C, n * p * sizeof(int) / 4, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_mcore);

    return t;
}
