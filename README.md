# Float8 Matrix Multiplication Simulation on Devices without TensorCore for FP8

This repository contains a straightforward implementation of matrix multiplication for FP8 on devices that lack
TensorCore support. It simulates the behavior of typical FP8 matrix multiplication (including all rounding
operations) using CUDA.

The current implementation uses a 32x8 block size and 4-item vectorization (to be implemented; presently, scalar
access is used for 4 items). It'a naive implementation and can be optimized further, but already achieves a reasonable
speed in my test environment.

## Purpose

The purpose of this project is to simulate the behavior of FP8 matrix multiplication on devices that do not support FP8
(pre-Ampere architecture) using CUDA.

Given that the rounding behavior of FP8 is significantly different from FP32, it is crucial to simulate FP8 matrix
multiplication to develop better algorithms for quantization and low-precision inference. However, due to the lack
of FP8 support on CPUs (and their limited performance) and older GPUs, engineers without access to newer architectures
like Hopper face challenges in developing FP8 algorithms.

This implementation offers a way to simulate the behavior of FP8 matrix multiplication with reasonable performance on
older devices. In our test environment, this naive implementation achieves `140 GFLOPS` on `1/8` of the temporal slice of
an Nvidia A16 GPU, where the FP32 throughput is `560 GFLOPS`.

It can enable engineers to develop and test FP8 algorithms on older devices without FP8 support, like laptops and
personal computers, and then deploy them on newer devices with FP8 support.
