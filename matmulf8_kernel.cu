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


#include <cuda_runtime.h>
#include "matmulf8_kernel.cuh"


__device__ __forceinline__ int access_byte(const int* __restrict__ data, int i) {
    return data[i>>2] >> ((i&3)<<3) & 0xff;
}

__device__ __forceinline__ int add(int a, int b, const int* __restrict__ acore){
#ifdef DB
    int r = 0;
    if((a&0x80)^(b&0x80)) {
        if(bt > at){
            r = access_byte(acore, (bt << 7) + at);
        }else{
            r = access_byte(acore, (at << 7) + bt);
        }
        if(((b&0x7f) > (a&0x7f)) && (b&0x80)){
            r ^= 0x80;
        }
    } else {
        if(at <= bt){
            r = access_byte(acore, (at << 7) + bt);
        }else{
            r = access_byte(acore, (bt << 7) + at);
        }
        r |= a & 0x80;
    }
    return r;
#else
    return access_byte(acore, (b << 7) + a);
#endif
}

__device__ __forceinline__ int reduce_f8(int vec, const int* __restrict__ acore) {
    return add(add(add((vec >> 24) & 0xff, (vec >> 16) & 0xff, acore), (vec >> 8) & 0xff, acore), vec & 0xff, acore);
}

__device__ __forceinline__ int addv4(int a, int b, const int* __restrict__ acore) {
    int res = 0;
    // TODO: Use unpack PTX instruction
#pragma unroll
    for(int i = 0; i < 4; i++) {
        res |= add((a >> (i * 8)) & 0xff, (b >> (i * 8)) & 0xff, acore) << (i * 8);
    }
    return res;
}

// Do a 4 8bit fma, r[i] = a[i] * b[i] + c[i]
__device__ __forceinline__ int fma8v4(int a, int b, int c, int* __restrict__ acore, int* __restrict__ mcore) {
    int mlt = 0;
    // TODO: Use unpack PTX instruction
#pragma unroll
    for(int i = 0; i < 4; i++) {
        int a0 = (a >> (i * 8)) & 0xff;
        int b0 = (b >> (i * 8)) & 0xff;
        mlt |= access_byte(mcore, ((a0&0xff)<<7) + (b0&0xff)) << (i * 8);
    }
    mlt |= (a&0x80808080) ^ (b&0x80808080);
    return addv4(c, mlt, acore);
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
    const int tx = threadIdx.x, ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int mz = m / 4, pz = p / 4;

    int res = 0;
    int dres;
    int rs[4];
    __shared__ int ac[4096], mc[4096];
    __shared__ int As[32 * 9], Bs[32 * 9];
    // Load core
    for(int i = 0; i < 4096; i += 256) {
        ac[i + tid] = __ldca(acore + i + tid);
        mc[i + tid] = __ldca(mcore + i + tid);
    }
    __syncthreads();

    for(int i = 0; i < mz; i += 8) {
        As[v * 9 + u] = A[(x0 + v) * mz + i + u];
        Bs[v * 9 + u] = B[(y0*4 + v) * mz + i + u];
        __syncthreads();
        // rs is at (x0 + tx, y0 + ty)
        rs[0] = rs[1] = rs[2] = rs[3] = 0;
        for(int j = 0; j < 8; j++) {
            for(int k = 0; k < 4; k++){
                rs[k] = fma8v4(As[tx*9+j], Bs[(ty*4+k)*9+j], rs[k], ac, mc);
            }
        }
        dres = 0;
        for(int j = 0; j < 4; j++){
            dres |= reduce_f8(rs[j], ac) << (j*8);
        }
        res = addv4(res, dres, ac);
        __syncthreads();
    }

    C[x * pz + y] = res;
}