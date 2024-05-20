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

// Do a 4 8bit fma, r[i] = a[i] * b[i] + c[i]
__device__ int fma8(int a, int b, int c, int* __restrict__ acore, int* __restrict__ mcore) {
    int r = 0;
#pragma unroll
    for(int i = 0; i < 4; i++) {
        int a0 = (a >> (i * 8)) & 0xff;
        int b0 = (b >> (i * 8)) & 0xff;
        int c0 = (c >> (i * 8)) & 0xff;
        int m = mcore[((a0&0x7f)<<7) + (b0&0x7f)];
        int r = 0;
        m |= (a0&0x80) ^ (b0&0x80);
        if((m&0x80)^(c0&0x80)) {
            if((m&0x7f) == (c0&0x7f)){
                r = 0x00;
            }else if(m&0x80){
                // c0 - m
                if((c0&0x7f) > (m&0x7f)){
                    r = acore[((c0&0x7f) << 7) + (m&0x7f)];
                }else{
                    r = acore[((m&0x7f) << 7) + (c0&0x7f)];
                    r ^= 0x80;
                }
            }
        }
    }
}