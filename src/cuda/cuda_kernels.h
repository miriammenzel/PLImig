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

/// Fixed median kernel size
constexpr auto MEDIAN_KERNEL_SIZE = 5;
/// Number of CUDA Kernel threads used for kernel execution
constexpr auto CUDA_KERNEL_NUM_THREADS = 32;

__device__ void shellSort(float* array, unsigned int low, unsigned int high);

__global__ void medianFilterKernel(const float* image, int image_stride,
                                   float* result_image, int result_image_stride,
                                   int2 imageDims);

__global__ void medianFilterMaskedKernel(const float* image, int image_stride,
                                         float* result_image, int result_image_stride,
                                         const unsigned char* mask, int mask_stride,
                                         int2 imageDims);

//// NEW CONNECTED COMPONENTS ALGORITHM
__global__ void connectedComponentsUFLocalMerge(cudaTextureObject_t inputTexture, unsigned int image_width, unsigned int image_height,
    unsigned int* labelMap, unsigned int label_stride);
__global__ void connectedComponentsUFGlobalMerge(cudaTextureObject_t inputTexture, unsigned int image_width, unsigned int image_height,
    unsigned int* labelMap, unsigned int label_stride);
__global__ void connectedComponentsUFPathCompression(cudaTextureObject_t inputTexture, unsigned int image_width, unsigned int image_height,
    unsigned int* labelMap, unsigned int label_stride);
__device__ void connectedComponentsUFUnion(unsigned int* L, unsigned int a, unsigned int b);
__device__ unsigned int connectedComponentsUFFind(const unsigned int* L, unsigned int index);

//// OLD CONNECTED COMPONENTS ALGORITHM
__global__ void connectedComponentsInitializeMask(const unsigned char* image, int image_stride,
    unsigned int* mask, int mask_stride, int line_width);
__global__ void connectedComponentsIteration(unsigned int* mask, int mask_stride, int2 maskDims, volatile bool* changeOccured);
__global__ void connectedComponentsReduceComponents(unsigned int* mask, int mask_stride,
                                                    const unsigned int* lutKeys, unsigned int lutSize);

__global__ void histogram(const float* image, int image_width, int image_height, unsigned int* histogram, float minVal, float maxVal, unsigned int numBins);
__global__ void histogramSharedMem(const float* image, int image_width, int image_height, unsigned int* histogram, float minVal, float maxVal, unsigned int numBins);

#endif //PLIMIG_CUDA_KERNELS_H
