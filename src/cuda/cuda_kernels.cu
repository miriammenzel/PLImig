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

__device__ void shellSort(float* array, unsigned int low, unsigned int high) {
    // Using the Ciura, 2001 sequence for best performance
    unsigned int gaps[8] = {1, 4, 10, 23, 57, 132, 301, 701};
    if(low < high) {
        float* subArr = array + low;
        unsigned int n = high - low;
        for (int pos = 7; pos > 0; --pos) {
            unsigned int gap = gaps[pos];
            // Do a gapped insertion sort for this gap size.
            // The first gap elements a[0..gap-1] are already in gapped order
            // keep adding one more element until the entire array is
            // gap sorted
            for (unsigned int i = gap; i < n; i += 1) {
                // add a[i] to the elements that have been gap sorted
                // save a[i] in temp and make a hole at position i
                float temp = subArr[i];

                // shift earlier gap-sorted elements up until the correct
                // location for a[i] is found
                unsigned int j;
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
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // The valid values will be counted to ensure that the median will be calculated correctly
    unsigned int validValues = 0;
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
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // The valid values will be counted to ensure that the median will be calculated correctly
    unsigned int validValues = 0;
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
        } else {
            result_image[x + y * result_image_stride] = buffer[0];
        }
    }
}

__global__ void connectedComponentsInitializeMask(const unsigned char* image, int image_stride,
                                                  unsigned int* mask, int mask_stride,
                                                  int line_width) {
    // Calculate actual position in image based on thread number and block number
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    mask[x + y * mask_stride] = (image[x + y * image_stride] & 1) * (y * (unsigned int) line_width + x + 1);
}

__global__ void connectedComponentsIteration(unsigned int* mask, int mask_stride, int2 maskDims, volatile bool* changeOccured) {
    // Calculate actual position in image based on thread number and block number
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int pixelVals[5];

    unsigned int maxVal = 0;
    if(mask[x + y * mask_stride] != 0) {
        pixelVals[0] = mask[x + y * mask_stride];
        pixelVals[1] = x-1 >= 0 ? mask[x-1 + y * mask_stride] : 0;
        pixelVals[2] = x+1 < maskDims.x ? mask[x+1 + y * mask_stride] : 0;
        pixelVals[3] = y-1 >= 0 ? mask[x + (y-1) * mask_stride] : 0;
        pixelVals[4] = y+1 < maskDims.y ? mask[x + (y+1) * mask_stride] : 0;

        #pragma unroll
        for(unsigned int i = 0; i < 5; ++i) {
            maxVal = max(maxVal, pixelVals[i]);
        }

        if(maxVal > mask[x + y * mask_stride]) {
            mask[x + y * mask_stride] = maxVal;
            *changeOccured = true;
        }
    }
}

__global__ void connectedComponentsReduceComponents(unsigned int* mask, int mask_stride,
                                                    const unsigned int* lutKeys,
                                                    const unsigned int lutSize) {
    // Calculate actual position in image based on thread number and block number
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (unsigned int i = 0; i < lutSize; ++i) {
        if(mask[x + y * mask_stride] == lutKeys[i]) {
            mask[x + y * mask_stride] = i;
            break;
        }
    }
}

__device__ unsigned int connectedComponentsUFFind(const unsigned int* L, unsigned int index) {
    unsigned int label = L[index];
    assert(label > 0);
    while (label - 1 != index) {
        index = label - 1;
        label = L[index];
        assert(label > 0);
    }
    return index;
}

__device__ void connectedComponentsUFUnion(unsigned int* L, unsigned int a, unsigned int b) {
    bool done;
    do {
        a = connectedComponentsUFFind(L, a);
        b = connectedComponentsUFFind(L, b);
        if(a < b) {
            unsigned int old = atomicMin(L + b, a + 1);
            done = (old == b + 1);
            b = old - 1;
        } else if(b < a) {
            unsigned int old = atomicMin(L + a, b + 1);
            done = (old == a + 1);
            a = old - 1;
        } else {
            done = true;
        }
    } while(!done);
}

__global__ void connectedComponentsUFLocalMerge(cudaTextureObject_t inputTexture, unsigned int image_width, unsigned int image_height,
    unsigned int* labelMap, unsigned int label_stride) {
    unsigned global_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned local_index = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ unsigned int labelSM[CUDA_KERNEL_NUM_THREADS * CUDA_KERNEL_NUM_THREADS];
    __shared__ unsigned char inputBuffer[CUDA_KERNEL_NUM_THREADS * CUDA_KERNEL_NUM_THREADS];

    // Initialize shared memory.
    // The first labels will just match the index. Those labels will later be changed depending on the algorithm
    // InputBuffer is just a shared memory copy of the relevant image information
    labelSM[local_index] = local_index + 1;
    inputBuffer[local_index] = tex2D<unsigned char>(inputTexture, float(global_x), float(global_y));
    __syncthreads();

    unsigned char imageValue = inputBuffer[local_index];
    if (global_x < image_width && global_y < image_height) {
        // Check four way connectivity
        if(imageValue) {
            if(threadIdx.x > 0 && inputBuffer[local_index - 1]) {
                connectedComponentsUFUnion(labelSM, local_index, local_index - 1);
            }
            if(threadIdx.y > 0 && inputBuffer[local_index - blockDim.x]) {
                connectedComponentsUFUnion(labelSM, local_index, local_index - blockDim.x);
            }
        // Check eight way connectivity
        } else {
            if(threadIdx.y > 0 && inputBuffer[local_index - blockDim.x]) {
                if(threadIdx.x > 0 && inputBuffer[local_index - 1]) {
                    connectedComponentsUFUnion(labelSM, local_index - blockDim.x, local_index - 1);
                }
                if(threadIdx.x < blockDim.x - 1 && inputBuffer[local_index + 1]) {
                    connectedComponentsUFUnion(labelSM, local_index - blockDim.x, local_index + 1);
                }
            }
        }
    }
    __syncthreads();

    // Copy data from shared to global memory
    if (global_x < image_width && global_y < image_height) {
        if(inputBuffer[threadIdx.x + threadIdx.y * blockDim.x]) {
            unsigned f = connectedComponentsUFFind(labelSM, local_index);
            unsigned f_row = f / blockDim.x;
            unsigned f_col = f % blockDim.x;
            unsigned global_f = (blockIdx.y * blockDim.y + f_row) * label_stride + (blockIdx.x * blockDim.x + f_col);
            labelMap[global_x + global_y * label_stride] = global_f + 1;
        } else {
            labelMap[global_x + global_y * label_stride] = 0;
        }
    }
}

__global__ void connectedComponentsUFGlobalMerge(cudaTextureObject_t inputTexture, unsigned int image_width, unsigned int image_height,
                                                 unsigned int* labelMap, unsigned int label_stride) {
    unsigned global_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned label_index = global_x + global_y * label_stride;

    if(global_x < image_width && global_y < image_height) {
        if(tex2D<unsigned char>(inputTexture, global_x, global_y)) {
            if(global_x > 0 && threadIdx.x == 0 && tex2D<unsigned char>(inputTexture, global_x-1, global_y)) {
                connectedComponentsUFUnion(labelMap, label_index, label_index - 1);
            }
            if(global_y > 0 && threadIdx.y == 0 && tex2D<unsigned char>(inputTexture, global_x, global_y-1)) {
                connectedComponentsUFUnion(labelMap, label_index, label_index - label_stride);
            }
        } else {
            if(global_y > 0 && threadIdx.x == 0 && tex2D<unsigned char>(inputTexture, global_x, global_y-1)) {
                if(global_x > 0 && (threadIdx.x == 0 || threadIdx.y == 0) && tex2D<unsigned char>(inputTexture, global_x-1, global_y)) {
                    connectedComponentsUFUnion(labelMap, label_index - label_stride, label_index - 1);
                }
                if(global_x < image_width - 1 && (threadIdx.x == blockDim.x - 1 || threadIdx.y == 0) &&
                   tex2D<unsigned char>(inputTexture, global_x + 1, global_y)) {
                    connectedComponentsUFUnion(labelMap, label_index - label_stride, label_index + 1);
                }
            }
        }
    }
}

__global__ void connectedComponentsUFPathCompression(cudaTextureObject_t inputTexture, unsigned int image_width, unsigned int image_height,
                                                     unsigned int* labelMap, unsigned int label_stride) {
    unsigned global_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned label_index = global_x + global_y * label_stride;

    if(global_x < image_width && global_y < image_height) {
        if(tex2D<unsigned char>(inputTexture, global_x, global_y)) {
            labelMap[global_x + global_y * label_stride] = connectedComponentsUFFind(labelMap, label_index) + 1;
        }
    }
}

__global__ void histogram(const float* image, int image_width, int image_height, unsigned int* histogram, float minVal, float maxVal, unsigned int numBins) {
    // Calculate actual position in image based on thread number and block number
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const float binWidth = (float(maxVal) - float(minVal)) / float(numBins);
    if(x < image_width && y < image_height) {
        if(image[x + y * image_width] >= minVal && image[x + y * image_width] <= maxVal) {
            unsigned int bin = min((unsigned int) ((image[x + y * image_width] - minVal) / binWidth), numBins - 1);
            atomicAdd(&histogram[bin], (unsigned int) 1);
        }
    }
}

__global__ void histogramSharedMem(const float* image, int image_width, int image_height, unsigned int* histogram, float minVal, float maxVal, unsigned int numBins) {
    // Calculate actual position in image based on thread number and block number
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int locId = threadIdx.y*blockDim.x+threadIdx.x;
    const float binWidth = (float(maxVal) - float(minVal)) / float(numBins);

    extern __shared__ unsigned int localHistogram[];
    #pragma unroll
    for(unsigned i = locId; i < numBins; i += blockDim.x * blockDim.y) {
        localHistogram[i] = 0;
    }

    __syncthreads();

    if(x < image_width && y < image_height) {
        if(image[x + y * image_width] >= minVal && image[x + y * image_width] <= maxVal) {
            unsigned int bin = min((unsigned int) ((image[x + y * image_width] - minVal) / binWidth), numBins - 1);
            atomicAdd(&localHistogram[bin], (unsigned int) 1);
        }
    }

    __syncthreads();

    #pragma unroll
    for(unsigned i = locId; i < numBins; i += blockDim.x * blockDim.y) {
        atomicAdd(&histogram[i], localHistogram[i]);
    }
}