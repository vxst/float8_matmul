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
#include <cinttypes>
#include "matmulf8_kernel.cuh"

#define FLOAT8_EXP(x)  ((x >> 2) & 0x1F)
#define FLOAT8_MANT(x) (x & 0x83)
#define FLOAT8_CONSTRUCT(sign, exp, mant) ((sign << 7) | (exp << 2) | (mant))
#define FLOAT8_NAN 0x7E
#define FLOAT8_ISNAN(x) ((FLOAT8_EXP(x) == 0x1F) && (FLOAT8_MANT(x) != 0))

// TODO: Vectorize this function
__host__ __device__ __forceinline__ uint8_t float8_e5m2_add(uint8_t a, uint8_t b) {
    // Assume there is no NaN, it's (not?) useful for matmul
    // if (FLOAT8_ISNAN(a) || FLOAT8_ISNAN(b)) {
    //     return FLOAT8_NAN;
    // }

    if (a == 0) return b;
    if (b == 0) return a;

    // Extract sign, exponent and mantissa from inputs
    uint8_t exp_a = FLOAT8_EXP(a);
    int8_t mant_a = FLOAT8_MANT(a);

    uint8_t exp_b = FLOAT8_EXP(b);
    int8_t mant_b = FLOAT8_MANT(b);

    // Handle subnormal numbers (when exponent is 0)
    if (exp_a == 0) {
        exp_a = 1;
    } else {
        mant_a |= 0x04;
    }

    if (exp_b == 0) {
        exp_b = 1;
    } else {
        mant_b |= 0x04;
    }

    // Align exponents
    int shift;
    if (exp_a > exp_b) {
        shift = exp_a - exp_b;
        exp_b = exp_a;
        mant_b >>= shift;
    } else if (exp_b > exp_a) {
        shift = exp_b - exp_a;
        exp_a = exp_b;
        mant_a >>= shift;
    }

    uint8_t result_sign, result_exp = exp_a;
    uint8_t result_mant;
    result_mant = mant_a + mant_b;
    result_sign = result_mant >> 7;
    result_mant &= 0x7F;

    // Normalize result
    if (result_mant >= 0x08) {
        result_mant >>= 1;
        result_exp += 1;
    }

    // Mask out the implicit leading 1, add only need once
    result_mant &= 0x03;

    // Check for overflow into NaN range
    // if (result_exp > 0x1F) {
    //     return FLOAT8_NAN;
    // }
    if (result_exp == 1 && result_mant < 4) {
        result_exp = 0;
    }
    uint8_t result = FLOAT8_CONSTRUCT(result_sign, result_exp, result_mant);
    // if ((result & 0x7F) >= 0x7C) {
    //     result = (result&0x80) | 0x7C;
    // }

    return result;
}

__device__ __host__ __forceinline__ int add(int a, int b){
    return float8_e5m2_add(a, b);
}

__device__ __forceinline__ int reduce_f8(int vec) {
    return add(add(add((vec >> 24) & 0xff, (vec >> 16) & 0xff), (vec >> 8) & 0xff), vec & 0xff);
}

#ifdef TEST
__device__ __host__ int addv4(int a, int b) {
#else
__device__ __forceinline__ int addv4(int a, int b) {
#endif
    int res = 0;
    // TODO: Use unpack PTX instruction
#pragma unroll
    for(int i = 0; i < 4; i++) {
        res |= add((a >> (i * 8)) & 0xff, (b >> (i * 8)) & 0xff) << (i * 8);
    }
    return res;
}

__device__ __host__ __forceinline__ uint8_t float8_e5m2_mlt(uint8_t a, uint8_t b) {
    // This LUT can fit in RF
    // WARNING: Rounding is different, to be more efficient
    // Only different when abs(x_f8_1 - x) == abs(x_f8_2 - x)
    if(a == 0 || b == 0) return 0;
    uint8_t a_m2 = a & 0b11 | 0b100;
    uint8_t b_m2 = b & 0b11 | 0b100;
    int8_t a_e5 = (a >> 2) & 0b11111;
    int8_t b_e5 = (b >> 2) & 0b11111;

    uint8_t m2 = (a_m2 * b_m2) >> 2;
    int8_t e5 = a_e5 + b_e5 - 15;
    if(m2 & 0b1000){
        e5++; m2 >>= 1;
    }
    if(e5 < 0)
        return 0;
    m2 &= 0b0011;

    return ((a&0x80)^(b&0x80)) | (e5 << 2) | m2;
}
// Below is the vectorized version of float8_e5m2_mlt
// Nvidia Doc:
// To use these functions you do not need to include any additional header files in your program.
// However, nvcc with error:
// error: identifier "__dp4a" is undefined
// And
// calling a __device__ function("__dp4a(int, int, int)") from a __host__ __device__ function is not allowed
// Perhaps it's a bug of my nvcc or my usage(?), currently I have to use the scalar version
// 
// __device__ __forceinline__ int float8_e5m2_vmlt(int a, int b) {
//     int a_m = a & 0x03030303, b_m = b & 0x03030303;
//     int a_e = (a & 0x7C7C7C7C) >> 2, b_e = (b & 0x7C7C7C7C) >> 2;
//     int e = a_e + b_e - 0x0F0F0F0F;
//     int m = __dp4a(a_m, b_m, 0);
//     int md = m >> 1;
//     int m_msk = (m & 0x08080808) >> 3;
//     e += m_msk;
//     int res = e | ((a & 0x80808080) ^ (b & 0x80808080));
//     if(m_msk & 0x00000001){
//         res |= md & 0x00000003;
//     }else{
//         res |= m & 0x00000003;
//     }
//     if(m_msk & 0x00000100){
//         res |= md & 0x00000300;
//     }else{
//         res |= m & 0x00000300;
//     }
//     if(m_msk & 0x00010000){
//         res |= md & 0x00030000;
//     }else{
//         res |= m & 0x00030000;
//     }
//     if(m_msk & 0x01000000){
//         res |= md & 0x03000000;
//     }else{
//         res |= m & 0x03000000;
//     }
//     return res;
// }

// Do a 4 8bit fma, r[i] = a[i] * b[i] + c[i]
#ifdef TEST
__device__ __host__ int fma8v4(int a, int b, int c) {
#else
__device__ __forceinline__ int fma8v4(int a, int b, int c) {
#endif
    // int mlt = float8_e5m2_vmlt(a, b);
    int mlt = 0;
#pragma unroll
    for(int i = 0; i < 4; i++) {
        int a0 = (a >> (i * 8)) & 0xff;
        int b0 = (b >> (i * 8)) & 0xff;
        mlt |= float8_e5m2_mlt(a0, b0) << (i * 8);
    }
    return addv4(c, mlt);
}

// A 32x8 core, do 32x32 matrix multiplication
// tx: 0-31, ty: 0-7
__global__ void matmulf8(int* __restrict__ A, int* __restrict__ B, int* __restrict__ C,
                         int n, int m, int p){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = blockIdx.x * blockDim.x, y0 = blockIdx.y * blockDim.y;
    const int u = threadIdx.x % 8, v = threadIdx.y * 4 + threadIdx.x / 8; // u: 0-7, v: 0-31, u is continuous
    const int tx = threadIdx.x, ty = threadIdx.y;
    int mz = m / 4, pz = p / 4;

    int res = 0;
    int dres;
    int rs[4];
    __shared__ int As[32 * 9], Bs[32 * 9];

    for(int i = 0; i < mz; i += 8) {
        As[v * 9 + u] = A[(x0 + v) * mz + i + u];
        Bs[v * 9 + u] = B[(y0*4 + v) * mz + i + u];
        __syncthreads();
        // rs is at (x0 + tx, y0 + ty)
        rs[0] = rs[1] = rs[2] = rs[3] = 0;
        for(int j = 0; j < 8; j++) {
            for(int k = 0; k < 4; k++){
                rs[k] = fma8v4(As[tx*9+j], Bs[(ty*4+k)*9+j], rs[k]);
            }
        }
        dres = 0;
        for(int j = 0; j < 4; j++){
            dres |= reduce_f8(rs[j]) << (j*8);
        }
        res = addv4(res, dres);
        __syncthreads();
    }

    C[x * pz + y] = res;
}