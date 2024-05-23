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
each matrix multiplication. This project is also an order of magnitude faster than IntelLabs' `FP8-Emulation-Toolkit`.
The current format is `float8_e5m2`, but other variations can be introduced relatively easily.

## Speed

This implementation offers a way to simulate the behavior of FP8 matrix multiplication with reasonable performance on
older devices. In my test environment, where the Tensor Core FP32 throughput is `560 GFLOPS`(`1/8 temporal slice`
of `A16`), the performance will be between `160GFLOPS` to `400GFLOPS` exculding shared memory swap(which doesn't
happen in MIG or real GPU), as Nsights reports 95% of the time is spent on Mio throttling if two matrices are
randomized, so theory performance is around `300GFLOPS`, (`19.1GFLOPS` with swapping and about `95%` time Mio related
throttling waiting for swapping), with a full zero matrix, the performance on `1/8` temporal slice is `166GFLOPS`,
since SASS code indicates the shared memory is always accessed(there is no layer between L1 and RF by design),
this should be the minimal performance on real world as full zero matrix is the worst case for shared memory
bank conflict. The performance for random matrix on a device with real shared memory should be 25% to 50% of
its FP32 performance(depends on the specific architecture).

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