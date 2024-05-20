// Copyright (C) 2024 Chunqing Shan
// 
// f8matmul is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// qrand is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with qrand. If not, see <http://www.gnu.org/licenses/>.


#include <cuda_runtime.h>


__device__ __forceinline__ int access_byte(int* __restrict__ data, int i) {
    return data[i>>2] >> ((i&3)<<3) & 0xff;
}

// Do a 4 8bit fma, r[i] = a[i] * b[i] + c[i]
__device__ __forceinline__ int fma8v4(int a, int b, int c, int* __restrict__ acore, int* __restrict__ mcore) {
    int res = 0;
#pragma unroll
    for(int i = 0; i < 4; i++) {
        int a0 = (a >> (i * 8)) & 0xff;
        int b0 = (b >> (i * 8)) & 0xff;
        int c0 = (c >> (i * 8)) & 0xff;
        int m = access_byte(mcore, ((a0&0x7f)<<7) + (b0&0x7f));
        int r = 0;
        m |= (a0&0x80) ^ (b0&0x80);
        if((m&0x80)^(c0&0x80)) {
            if((m&0x7f) == (c0&0x7f)){
                r = 0x00;
            }else if(m&0x80){
                // c0 - m
                if((c0&0x7f) > (m&0x7f)){
                    r = access_byte(acore, ((c0&0x7f) << 7) + (m&0x7f));
                }else{
                    r = access_byte(acore, ((m&0x7f) << 7) + (c0&0x7f));
                    r ^= 0x80;
                }
            }else{
                // m - c0
                if((m&0x7f) > (c0&0x7f)){
                    r = access_byte(acore, ((m&0x7f) << 7) + (c0&0x7f));
                }else{
                    r = access_byte(acore, ((c0&0x7f) << 7) + (m&0x7f));
                    r ^= 0x80;
                }
            }
        } else {
            if((m&0x7f) <= (c0&0x7f)){
                r = access_byte(acore, ((m&0x7f) << 7) + (c0&0x7f));
            }else{
                r = access_byte(acore, ((c0&0x7f) << 7) + (m&0x7f));
            }
            r |= m & 0x80;
        }
        res |= r << (i * 8);
    }
    return res;
}

// A 32x8 core, do 32x32 matrix multiplication
// tx: 0-31, ty: 0-7
__global__ void matmulf8(int* __restrict__ A, int* __restrict__ B, int* __restrict__ C,
                         int n, int m, int p,
                         int* __restrict__ acore, int* __restrict__ mcore) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = blockIdx.x * blockDim.x, y0 = blockIdx.y * blockDim.y;
    const int u = threadIdx.x % 8, v = threadIdx.y * 4 + threadIdx.x / 8; // u: 0-7, v: 0-31, u is continuous
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int mz = m / 4, pz = p / 4;

    int res = 0;
    __shared__ int ac[4096], mc[4096];
    __shared__ int As[32 * 8], Bs[4][32 * 8];
    // Load core
    for(int i = 0; i < 4096; i += 256) {
        ac[i + tid] = __ldcg(acore + i + tid);
        mc[i + tid] = __ldcg(mcore + i + tid);
    }
    __syncthreads();

    for(int i = 0; i < m; i += 32) {
        As[v * 32 + u] = A[(x0 + v) * mz + i + u];
#pragma unroll
        for(int j = 0; j < 4; j++) {
            Bs[j][v * 32 + u] = B[(y0 * 4 + j) * mz + i + u];
        }
    }
    __syncthreads();

    C[x * pz + y] = res;
}