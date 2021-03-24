/*
    MIT License

    Copyright (c) 2021 Forschungszentrum Jülich / Jan André Reuter.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
 */

#ifndef PLIMIG_CUDA_KERNELS_H
#define PLIMIG_CUDA_KERNELS_H

#include <cuda.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <cstdio>

/// Fixed median kernel size
#define MEDIAN_KERNEL_SIZE 10
/// Number of CUDA Kernel threads used for kernel execution
#define CUDA_KERNEL_NUM_THREADS 32

__device__ void shellSort(float* array, uint low, uint high);

__global__ void medianFilterKernel(const float* image, int image_stride,
                                   float* result_image, int result_image_stride,
                                   int2 imageDims);

__global__ void medianFilterMaskedKernel(const float* image, int image_stride,
                                         float* result_image, int result_image_stride,
                                         const unsigned char* mask, int mask_stride,
                                         int2 imageDims);

//// NEW CONNECTED COMPONENTS ALGORITHM
__global__ void connectedComponentsUFLocalMerge(cudaTextureObject_t inputTexture, uint image_width, uint image_height,
                                                uint* labelMap, uint label_stride);
__global__ void connectedComponentsUFGlobalMerge(cudaTextureObject_t inputTexture, uint image_width, uint image_height,
                                                uint* labelMap, uint label_stride);
__global__ void connectedComponentsUFPathCompression(cudaTextureObject_t inputTexture, uint image_width, uint image_height,
                                                uint* labelMap, uint label_stride);
__device__ void connectedComponentsUFUnion(uint* L, uint a, uint b);
__device__ uint connectedComponentsUFFind(uint* L, uint index);

//// OLD CONNECTED COMPONENTS ALGORITHM
__global__ void connectedComponentsInitializeMask(const unsigned char* image, int image_stride,
                                                  uint* mask, int mask_stride,
                                                  int line_width);
__global__ void connectedComponentsIteration(uint* mask, int mask_stride, int2 maskDims, volatile bool* changeOccured);
__global__ void connectedComponentsReduceComponents(uint* mask, int mask_stride,
                                                    const uint* lutKeys,
                                                    uint lutSize);

__global__ void histogram(uint* image, int image_width, int image_height, uint* histogram, uint min, uint max);
__global__ void histogramSharedMem(uint* image, int image_width, int image_height, uint* histogram, uint min, uint max);

#endif //PLIMIG_CUDA_KERNELS_H
