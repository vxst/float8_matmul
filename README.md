# High performance CUDA FP8 Matrix Multiplication Simulation on Devices without FP8 Support

This is a high performance implementation of matrix multiplication for FP8 on devices that lack
TensorCore support. It simulates the behavior of typical FP8 matrix multiplication (including all accumulation behavior)
using general CUDA.

The current implementation uses a 32x8 block size and 4-item vectorization (to be implemented; presently, scalar
access is used for 4 items). It's a naive implementation and can be optimized further, but already achieves a reasonable
speed in my test environment.


## Purpose

The purpose of this project is to simulate the behavior of FP8 matrix multiplication on devices that do not support FP8
(pre-Ampere architecture) using CUDA.

As the accumulation behavior of FP8 is significantly different from FP32, it is important for engineers to be able to simulate FP8 matrix
multiplication to develop better algorithms for quantization and low-precision inference. However, due to the lack
of FP8 support on CPUs and older GPUs, engineers without access to latest GPU architectures
like Hopper face challenges in developing FP8 algorithms(actually impossible since the accumulation behavior is very
different, and people really can't refine a real world neural network with CPU simulation).

## Compare

Currently, there is no existing project that simulates FP8 matrix multiplication with FP8 accumulation behavior. 
The [FP8-Emulation-Toolkit](https://github.com/IntelLabs/FP8-Emulation-Toolkit) only performs rounding to FP8 after(?really??)
each matrix multiplication. The current format is `float8_e5m2`, but other variations can be introduced relatively easily.

## Speed

Since I have no hardware to know whether LUT is faster than calc in real hardware(95% throllte when run with 1/8
of A16, normally shared memory throllte should be below 60% even for extensive use), I have implemented add with
calc and kept mlt with LUT. The add implementation is very naive and without any vectorization, normally it can
be 4x faster with vectorization. Since my hardware throllte on any intensive shared memory access, I'd rather
keep it simple and optimize it later.

Still, the performance is the same order of magnitude as the FP32 matrix multiplication, much faster than FP64,
on `1/8` A16, FP64 performance is `17.5GFLOPS`, while this implementation is `124.6GFLOPS`, so it might be good
enough for simulation purposes.

It can enable engineers to develop and test FP8 algorithms on older devices without FP8 support, like laptops and
personal computers, and then deploy them on newer devices with FP8 support.

# WARNING: THIS IS AN EXPERIMENTAL PROJECT, NOT FULLY TESTED YET

```
Copyright (C) 2024 Chunqing Shan

float8_matmul is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

float8_matmul is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with float8_matmul. If not, see <http://www.gnu.org/licenses/>.
```