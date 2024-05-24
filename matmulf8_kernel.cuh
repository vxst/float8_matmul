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


#pragma once

#include <cuda_runtime.h>

#ifdef TEST
__device__ __host__ int addv4(int a, int b);
__device__ __host__ int fma8v4(int a, int b, int c, int* __restrict__ mcore);
#endif

__global__ void matmulf8(int* __restrict__ A, int* __restrict__ B, int* __restrict__ C,
                         int n, int m, int p,
                         int* __restrict__ mcore);