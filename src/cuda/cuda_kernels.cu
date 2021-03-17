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

#include "cuda_kernels.h"

__device__ void shellSort(float* array, uint low, uint high) {
    // Using the Ciura, 2001 sequence for best performance
    uint gaps[8] = {1, 4, 10, 23, 57, 132, 301, 701};
    if(low < high) {
        float* subArr = array + low;
        uint n = high - low;
        for (int pos = 7; pos > 0; --pos) {
            uint gap = gaps[pos];
            // Do a gapped insertion sort for this gap size.
            // The first gap elements a[0..gap-1] are already in gapped order
            // keep adding one more element until the entire array is
            // gap sorted
            for (uint i = gap; i < n; i += 1) {
                // add a[i] to the elements that have been gap sorted
                // save a[i] in temp and make a hole at position i
                float temp = subArr[i];

                // shift earlier gap-sorted elements up until the correct
                // location for a[i] is found
                uint j;
                for (j = i; j >= gap && subArr[j - gap] > temp; j -= gap) {
                    subArr[j] = subArr[j - gap];
                }

                // put temp (the original a[i]) in its correct location
                subArr[j] = temp;
            }
        }
    }
}

__global__ void medianFilterKernel(const float* image, int image_stride,
                                   float* result_image, int result_image_stride,
                                   int2 imageDims) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // The valid values will be counted to ensure that the median will be calculated correctly
    uint validValues = 0;
    int cy_bound;
    // Median filter buffer
    float buffer[4 * MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];

    // Only try to calculate the median of pixels within the non-padded image
    if(x >= MEDIAN_KERNEL_SIZE && x < imageDims.x - MEDIAN_KERNEL_SIZE && y >= MEDIAN_KERNEL_SIZE && y < imageDims.y - MEDIAN_KERNEL_SIZE) {
        // Transfer image pixels to our kernel for median filtering application
        for (int cx = -MEDIAN_KERNEL_SIZE; cx < MEDIAN_KERNEL_SIZE; ++cx) {
            // The median filter kernel is round. Therefore calculate the valid y-positions based on our x-position in the kernel
            cy_bound = sqrtf(MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE - cx * cx);
            for (int cy = -cy_bound; cy <= cy_bound; ++cy) {
                // Save values in buffer
                buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                ++validValues;
            }
        }
        shellSort(buffer, 0, validValues);
        // Get middle value as our median
        result_image[x + y * result_image_stride] = buffer[validValues / 2];
    }
}

__global__ void medianFilterMaskedKernel(const float* image, int image_stride,
                                         float* result_image, int result_image_stride,
                                         const unsigned char* mask, int mask_stride,
                                         int2 imageDims) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // The valid values will be counted to ensure that the median will be calculated correctly
    uint validValues = 0;
    int cy_bound;
    // Median filter buffer
    float buffer[4 * MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];

    // Only try to calculate the median of pixels within the non-padded image
    if(x >= MEDIAN_KERNEL_SIZE && x < imageDims.x - MEDIAN_KERNEL_SIZE && y >= MEDIAN_KERNEL_SIZE && y < imageDims.y - MEDIAN_KERNEL_SIZE) {
        // Transfer image pixels to our kernel for median filtering application
        for (int cx = -MEDIAN_KERNEL_SIZE; cx < MEDIAN_KERNEL_SIZE; ++cx) {
            // The median filter kernel is round. Therefore calculate the valid y-positions based on our x-position in the kernel
            cy_bound = sqrtf(MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE - cx * cx);
            for (int cy = -cy_bound; cy <= cy_bound; ++cy) {
                // If the pixel in the kernel matches the current pixel on the gray / white mask
                if (mask[x + y * mask_stride] == mask[x + cx + (y + cy) * image_stride]) {
                    // Save values in buffer
                    buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                    ++validValues;
                }
            }
        }
        // Depending on the number of valid values, calculate the median, save the pixel value itself or save zero
        if (validValues > 1) {
            shellSort(buffer, 0, validValues);
            result_image[x + y * result_image_stride] = buffer[validValues / 2];
        } else if (validValues == 1) {
            result_image[x + y * result_image_stride] = buffer[0];
        } else {
            result_image[x + y * result_image_stride] = 0;
        }
    }
}

__global__ void connectedComponentsInitializeMask(const unsigned char* image, int image_stride,
                                                  uint* mask, int mask_stride,
                                                  int line_width) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(image[x + y * image_stride] != 0) {
        mask[x + y * mask_stride] = y * uint(line_width) + x + 1;
    } else {
        mask[x + y * mask_stride] = 0;
    }
}

__global__ void connectedComponentsIteration(uint* mask, int mask_stride, int2 maskDims, volatile bool* changeOccured) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint minVal;
    if(mask[x + y * mask_stride] != 0) {
        minVal = mask[x + y * mask_stride];

        if(int(x - 1) >= 0 && mask[x-1 + y * mask_stride] != 0) {
            minVal = min(minVal, mask[x-1 + y * mask_stride]);
        }
        if(int(x + 1) < maskDims.x && mask[x+1 + y * mask_stride] != 0) {
            minVal = min(minVal, mask[x+1 + y * mask_stride]);
        }
        if(int(y - 1) >= 0 && mask[x + (y-1) * mask_stride] != 0) {
            minVal = min(minVal, mask[x + (y-1) * mask_stride]);
        }
        if(int(y + 1) < maskDims.y && mask[x + (y+1) * mask_stride] != 0) {
            minVal = min(minVal, mask[x + (y+1) * mask_stride]);
        }

        if(minVal != mask[x + y * mask_stride]) {
            mask[x + y * mask_stride] = minVal;
            *changeOccured = true;
        }
    }
}

__global__ void connectedComponentsReduceComponents(uint* mask, int mask_stride,
                                                    const uint* lutKeys,
                                                    const uint lutSize) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    for (uint i = 0; i < lutSize; ++i) {
        if(mask[x + y * mask_stride] == lutKeys[i]) {
            mask[x + y * mask_stride] = i;
            break;
        }
    }
}

__global__ void histogram(uint* image, int image_width, int image_height, uint* histogram, uint min, uint max) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint pixelValue = image[x + y * image_width];
    if(x > 0 && x < image_width && y > 0 && y < image_height) {
        if(pixelValue < max && pixelValue > min) {
            atomicAdd(&histogram[pixelValue - min], uint(1));
        }
    }
}