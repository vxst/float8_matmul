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


#include <cstdio>
#include <cuda_runtime.h>
#include "load_core.cuh"

int* load_core(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if(f == NULL) {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return NULL;
    }
    int* data;
    cudaMallocHost(&data, 16384);
    fread(data, sizeof(int), 16384 / sizeof(int), f);
    fclose(f);
    return data;
}
