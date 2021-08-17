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

#include "cuda/cuda_toolbox.h"

cv::Mat PLImg::cuda::raw::labeling::CUDAConnectedComponents(const cv::Mat& image, uint* maxLabelNumber) {
    // Prepare image for CUDA kernel
    cv::Mat kernelImage;
    // 1. Convert it to 8 bit unsigned integer values
    image.convertTo(kernelImage, CV_8UC1);
    // 2. Check if the image needs padding to allow the execution of our CUDA kernel
    uint heightPadding = CUDA_KERNEL_NUM_THREADS - kernelImage.rows % CUDA_KERNEL_NUM_THREADS;
    uint widthPadding = CUDA_KERNEL_NUM_THREADS - kernelImage.cols % CUDA_KERNEL_NUM_THREADS;
    cv::copyMakeBorder(kernelImage, kernelImage, heightPadding, 0, widthPadding, 0, cv::BORDER_CONSTANT, 0);

    // Create output resulting image for our needs
    cv::Mat result = cv::Mat(kernelImage.rows, kernelImage.cols, CV_32SC1);

    uchar* deviceImage;
    uint* deviceMask;
    bool* deviceChangeOccured;
    bool changeOccured;

    CHECK_CUDA(cudaMalloc(&deviceImage, kernelImage.total() * sizeof(uchar)));
    CHECK_CUDA(cudaMemcpy(deviceImage, kernelImage.data, kernelImage.total() * sizeof(uchar), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&deviceMask, kernelImage.total() * sizeof(uint)));
    CHECK_CUDA(cudaMalloc(&deviceChangeOccured, sizeof(bool)));

    dim3 threadsPerBlock, numBlocks;
    threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
    numBlocks = dim3(ceil(float(kernelImage.cols) / threadsPerBlock.x), ceil(float(kernelImage.rows) / threadsPerBlock.y));

    connectedComponentsInitializeMask<<<numBlocks, threadsPerBlock>>>(deviceImage, kernelImage.cols, deviceMask, kernelImage.cols, kernelImage.cols);
    CHECK_CUDA(cudaFree(deviceImage));
    do {
        CHECK_CUDA(cudaMemset(deviceChangeOccured, false, sizeof(bool)));
        connectedComponentsIteration<<<numBlocks, threadsPerBlock>>>(deviceMask, kernelImage.cols, {kernelImage.cols, kernelImage.rows},
                                                                     deviceChangeOccured);
        CHECK_CUDA(cudaMemcpy(&changeOccured, deviceChangeOccured, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(changeOccured);
    CHECK_CUDA(cudaFree(deviceChangeOccured));

    uint* deviceUniqueMask;
    CHECK_CUDA(cudaMalloc(&deviceUniqueMask, kernelImage.total() * sizeof(uint)));
    CHECK_CUDA(cudaMemcpy(deviceUniqueMask, deviceMask, kernelImage.total() * sizeof(uint), cudaMemcpyDeviceToDevice));
    thrust::sort(thrust::device, deviceUniqueMask, deviceUniqueMask + kernelImage.total());
    uint* deviceMaxUniqueLabel = thrust::unique(thrust::device, deviceUniqueMask, deviceUniqueMask + kernelImage.total());

    uint distance = thrust::distance(deviceUniqueMask, deviceMaxUniqueLabel);
    connectedComponentsReduceComponents<<<numBlocks, threadsPerBlock>>>(deviceMask, kernelImage.cols,
                                                                        deviceUniqueMask,
                                                                        distance);
    CHECK_CUDA(cudaFree(deviceUniqueMask));

    uint* deviceMaxLabel = thrust::max_element(thrust::device, deviceMask, deviceMask + kernelImage.total());
    CHECK_CUDA(cudaMemcpy(maxLabelNumber, deviceMaxLabel, sizeof(uint), cudaMemcpyDeviceToHost));

    // Copy result from GPU back to CPU
    CHECK_CUDA(cudaMemcpy(result.data, deviceMask, kernelImage.total() * sizeof(uint), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(deviceMask));
    CHECK_CUDA(cudaDeviceSynchronize());

    cv::Mat croppedImage = cv::Mat(result, cv::Rect(widthPadding, heightPadding, result.cols - widthPadding, result.rows - heightPadding));
    croppedImage.copyTo(result);

    return result;
}

cv::Mat PLImg::cuda::raw::labeling::CUDAConnectedComponentsUF(const cv::Mat &image, uint *maxLabelNumber) {
    // Prepare image for CUDA kernel
    cv::Mat kernelImage;
    // 1. Convert it to 8 bit unsigned integer values
    image.convertTo(kernelImage, CV_8UC1);
    // 2. Check if the image needs padding to allow the execution of our CUDA kernel
    uint heightPadding = CUDA_KERNEL_NUM_THREADS - kernelImage.rows % CUDA_KERNEL_NUM_THREADS;
    uint widthPadding = CUDA_KERNEL_NUM_THREADS - kernelImage.cols % CUDA_KERNEL_NUM_THREADS;
    cv::copyMakeBorder(kernelImage, kernelImage, heightPadding, 0, widthPadding, 0, cv::BORDER_CONSTANT, 0);

    // Create output resulting image for our needs
    cv::Mat result = cv::Mat(kernelImage.rows, kernelImage.cols, CV_32SC1);
    uint* deviceMask;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* imageArray;
    CHECK_CUDA(cudaMallocArray(&imageArray, &channelDesc, kernelImage.cols, kernelImage.rows));
    CHECK_CUDA(cudaMemcpy2DToArray(imageArray, 0, 0, kernelImage.data, kernelImage.cols * sizeof(uchar), kernelImage.cols * sizeof(uchar), kernelImage.rows, cudaMemcpyHostToDevice));

    // Step 1. Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = imageArray;
    // Step 2. Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Step 3: Create texture object
    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    CHECK_CUDA(cudaMalloc(&deviceMask, kernelImage.total() * sizeof(uint)));

    // Define CUDA kernel parameters
    dim3 threadsPerBlock, numBlocks;
    threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
    numBlocks = dim3(ceil(float(kernelImage.cols) / threadsPerBlock.x), ceil(float(kernelImage.rows) / threadsPerBlock.y));

    // First step. Do local connected components on each block
    connectedComponentsUFLocalMerge<<<numBlocks, threadsPerBlock>>>(texObj, kernelImage.cols, kernelImage.rows, deviceMask, kernelImage.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    // Second step. Fix lines between each block.
    connectedComponentsUFGlobalMerge<<<numBlocks, threadsPerBlock>>>(texObj, kernelImage.cols, kernelImage.rows, deviceMask, kernelImage.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    // Third step. Fix paths which might be wrong after the global merge
    connectedComponentsUFPathCompression<<<numBlocks, threadsPerBlock>>>(texObj, kernelImage.cols, kernelImage.rows, deviceMask, kernelImage.cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFreeArray(imageArray));

    // Fourth step. Reduce label numbers to reasonable numbers.
    uint* deviceUniqueMask;
    CHECK_CUDA(cudaMalloc(&deviceUniqueMask, kernelImage.total() * sizeof(uint)));
    CHECK_CUDA(cudaMemcpy(deviceUniqueMask, deviceMask, kernelImage.total() * sizeof(uint), cudaMemcpyDeviceToDevice));
    thrust::sort(thrust::device, deviceUniqueMask, deviceUniqueMask + kernelImage.total());
    uint* deviceMaxUniqueLabel = thrust::unique(thrust::device, deviceUniqueMask, deviceUniqueMask + kernelImage.total());
    // Save the new maximum label as a return value for the user
    uint distance = thrust::distance(deviceUniqueMask, deviceMaxUniqueLabel);
    if(maxLabelNumber) {
        *maxLabelNumber = distance;
    }
    // Reduce numbers in label image to low numbers for following algorithms
    connectedComponentsReduceComponents<<<numBlocks, threadsPerBlock>>>(deviceMask, kernelImage.cols,
                                                                        deviceUniqueMask,
                                                                        distance);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(deviceUniqueMask));
    CHECK_CUDA(cudaMemcpy(result.data, deviceMask, kernelImage.total() * sizeof(uint), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(deviceMask));

    cv::Mat croppedImage = cv::Mat(result, cv::Rect(widthPadding, heightPadding, result.cols - widthPadding, result.rows - heightPadding));
    croppedImage.copyTo(result);

    return result;
}

void PLImg::cuda::raw::filters::CUDAmedianFilter(cv::Mat& image, cv::Mat& result) {
    float* deviceImage, *deviceResult;
    int nSrcStep, nResStep;
    int2 subImageDims;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;
    // Allocate GPU memory for the original image and its result
    CHECK_CUDA(cudaMalloc((void **) &deviceImage, image.total() * image.elemSize()));
    // Length of columns
    nSrcStep = image.cols;

    CHECK_CUDA(cudaMalloc((void **) &deviceResult, image.total() * image.elemSize()));
    // Length of columns
    nResStep = result.cols;

    // Copy image from CPU to GPU
    CHECK_CUDA(cudaMemcpy(deviceImage, image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice));

    // Apply median filter
    subImageDims = {result.cols, result.rows};
    threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
    numBlocks = dim3(ceil(float(subImageDims.x) / threadsPerBlock.x), ceil(float(subImageDims.y) / threadsPerBlock.y));
    // Run median filter
    medianFilterKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                       deviceResult, nResStep,
                                                       subImageDims);

    // Copy result from GPU back to CPU
    CHECK_CUDA(cudaMemcpy(result.data, deviceResult, image.total() * image.elemSize(), cudaMemcpyDeviceToHost));

    // Free reserved memory
    CHECK_CUDA(cudaFree(deviceImage));
    CHECK_CUDA(cudaFree(deviceResult));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void PLImg::cuda::raw::filters::CUDAmedianFilterMasked(cv::Mat& image, cv::Mat& mask, cv::Mat& result) {
    float* deviceImage, *deviceResult;
    uchar* deviceMask;
    ulong nSrcStep, nMaskStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;
    int2 subImageDims;

    // Allocate GPU memory for the original image, mask and its result
    CHECK_CUDA(cudaMalloc((void **) &deviceImage, image.total() * image.elemSize()));
    // Length of columns
    nSrcStep = image.cols;

    CHECK_CUDA(cudaMalloc((void **) &deviceMask, mask.total() * mask.elemSize()));
    // Length of columns
    nMaskStep = mask.cols;

    CHECK_CUDA(cudaMalloc((void **) &deviceResult, image.total() * image.elemSize()));
    // Length of columns
    nResStep = result.cols;

    // Copy image from CPU to GPU
    CHECK_CUDA(cudaMemcpy(deviceImage, image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceMask, mask.data, mask.total() * mask.elemSize(), cudaMemcpyHostToDevice));

    // Apply median filter
    subImageDims = {image.cols, image.rows};
    threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
    numBlocks = dim3(ceil(float(subImageDims.x) / threadsPerBlock.x), ceil(float(subImageDims.y) / threadsPerBlock.y));
    // Run median filter
    medianFilterMaskedKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                             deviceResult, nResStep,
                                                             deviceMask, nMaskStep,
                                                             subImageDims);

    CHECK_CUDA(cudaMemcpy(result.data, deviceResult, image.total() * image.elemSize(), cudaMemcpyDeviceToHost));

    // Free reserved memory
    CHECK_CUDA(cudaFree(deviceImage));
    CHECK_CUDA(cudaFree(deviceResult));
    CHECK_CUDA(cudaFree(deviceMask));
    CHECK_CUDA(cudaDeviceSynchronize());
}

cv::Mat PLImg::cuda::raw::CUDAhistogram(const cv::Mat &image, float minLabel, float maxLabel, uint numBins) {
    float* deviceImage;
    uint* deviceHistogram;

    CHECK_CUDA(cudaMalloc(&deviceImage, image.total() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(deviceImage, image.data, image.total() * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&deviceHistogram, numBins * sizeof(uint)));
    CHECK_CUDA(cudaMemset(deviceHistogram, 0, numBins * sizeof(uint)));

    dim3 threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
    dim3 numBlocks = dim3(ceil(float(image.cols) / threadsPerBlock.x), ceil(float(image.rows) / threadsPerBlock.y));

    cv::Mat hostHistogram(numBins, 1, CV_32SC1);
    if(numBins * sizeof(uint) < 49152) {
        histogramSharedMem<<<numBlocks, threadsPerBlock, numBins * sizeof(uint)>>>
        (deviceImage, image.cols, image.rows, deviceHistogram, minLabel, maxLabel, numBins);
    } else {
        histogram<<<numBlocks, threadsPerBlock>>>
        (deviceImage, image.cols, image.rows, deviceHistogram, minLabel, maxLabel, numBins);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hostHistogram.data, deviceHistogram, numBins * sizeof(uint), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(deviceHistogram));
    CHECK_CUDA(cudaFree(deviceImage));

    return hostHistogram;
}

