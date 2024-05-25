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

Currently calculate the float8 rather than LUT, can be vectorized. The implementation is scalar, but should be
vectorizable, which may provide 4x speedup, since all bit manipulation and multiplication can be vectorized.

Still, the performance is the same order of magnitude as the FP32 matrix multiplication, faster than FP64,
on `1/8` A16, FP64 performance is `17.5GFLOPS`, while this implementation is `32.3GFLOPS`, so it might be good
enough for simulation purposes. After fully vectorized, the performance should be around `100GFLOPS` for `1/8` `A16`(4:1),
and it should be scalable on older devices(since shared memory usage is minimal). It can be further optimized by
using bigger kernel. The target is 4:1 to 2:1 performance ratio between FP32 and FP8, but even current performance
at 15:1, which should be more than `1TFLOPS` on `A100`, should be good enough for simulation/research purposes.

Vectorization is not online as NVIDIA documents[1] something but doesn't compile. There is PTX instruction though,
perhaps my usage is somehow wrong(?). Will look into it later.

It can enable engineers to develop and research on FP8 algorithms on older devices without FP8 support, but with
CUDA support, should be more accessible for general developer than Hoppper.

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

1. https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html