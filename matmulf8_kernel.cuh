#pragma once

#include <cuda_runtime.h>

__global__ void matmulf8(int* __restrict__ A, int* __restrict__ B, int* __restrict__ C,
                         int n, int m, int p,
                         int* __restrict__ acore, int* __restrict__ mcore);