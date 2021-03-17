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

    std::cout << numBlocks.x * threadsPerBlock.x << " " << numBlocks.y * threadsPerBlock.y << std::endl;
    std::cout << kernelImage.cols << " " << kernelImage.rows << std::endl;
    std::cout << int(heightPadding) << " " << int(widthPadding) << std::endl;

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

std::shared_ptr<cv::Mat> PLImg::cuda::raw::filters::CUDAmedianFilter(const std::shared_ptr<cv::Mat>& image) {
    // Create a result image with the same dimensions as our input image
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    // Expand borders of input image inplace to ensure that the median algorithm can run correcly
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // The image might be too large to be saved completely in the video memory.
    // Therefore chunks will be used if the amount of memory is too small.
    uint numberOfChunks = 1;
    // Check the free video memory
    ulong freeMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, nullptr));
    // If the total free memory is smaller than the estimated amount of memory, calculate the number of chunks
    // with the power of four (1, 4, 16, 256, 1024, ...)
    if(double(image->total()) * image->elemSize() * 2.1 > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(image->total() * image->elemSize() * 2.1 / double(freeMem)) / log(4))));
    }
    // Each dimensions will get the same number of chunks. Calculate them by using the square root.
    uint chunksPerDim = fmax(1, sqrtf(numberOfChunks));

    float* deviceImage, *deviceResult;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;

    // We've increased the image dimensions earlier. Save the original image dimensions for further calculations.
    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    int2 subImageDims;

    cv::Mat subImage, subResult, croppedImage;
    // For each chunk
    for(uint it = 0; it < numberOfChunks; ++it) {
        std::cout << "\rCurrent chunk: " << it+1 << "/" << numberOfChunks;
        std::flush(std::cout);
        // Calculate image boarders
        xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
        yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

        // Get chunk of our image and result. Apply padding to the result to ensure that the median filter will run correctly.
        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

        // Allocate GPU memory for the original image and its result
        CHECK_CUDA(cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize()));
        // Length of columns
        nSrcStep = subImage.cols;

        CHECK_CUDA(cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize()));
        // Length of columns
        nResStep = subResult.cols;

        // Copy image from CPU to GPU
        CHECK_CUDA(cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice));

        // Apply median filter
        subImageDims = {subImage.cols, subImage.rows};
        threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
        numBlocks = dim3(ceil(float(subImageDims.x) / threadsPerBlock.x), ceil(float(subImageDims.y) / threadsPerBlock.y));
        // Run median filter
        medianFilterKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                           deviceResult, nResStep,
                                                           subImageDims);

        // Copy result from GPU back to CPU
        CHECK_CUDA(cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost));

        // Free reserved memory
        CHECK_CUDA(cudaFree(deviceImage));
        CHECK_CUDA(cudaFree(deviceResult));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Calculate the range where the median filter was applied and where the chunk will be placed.
        cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, xMax - xMin, yMax - yMin);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        subResult(srcRect).copyTo(result(dstRect));
    }
    // Fix output after \r
    std::cout << std::endl;
    // Revert the padding of the original image
    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);

    // Return resulting median filtered image
    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::cuda::raw::filters::CUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask) {
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(*mask, *mask, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // The image might be too large to be saved completely in the video memory.
    // Therefore chunks will be used if the amount of memory is too small.
    uint numberOfChunks = 1;
    ulong freeMem;
    // Check the free video memory
    CHECK_CUDA(cudaMemGetInfo(&freeMem, nullptr));
    // If the total free memory is smaller than the estimated amount of memory, calculate the number of chunks
    // with the power of four (1, 4, 16, 256, 1024, ...)
    if(double(image->total()) * image->elemSize() * 3.1 > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(image->total() * image->elemSize() * 3.1 / double(freeMem)) / log(4))));
    }
    // Each dimensions will get the same number of chunks. Calculate them by using the square root.
    uint chunksPerDim = fmax(1, sqrtf(numberOfChunks));

    float* deviceImage, *deviceResult;
    uchar* deviceMask;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nMaskStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;

    // We've increased the image dimensions earlier. Save the original image dimensions for further calculations.
    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    int2 subImageDims;

    cv::Mat subImage, subMask, subResult, croppedImage;
    for(uint it = 0; it < numberOfChunks; ++it) {
        std::cout << "\rCurrent chunk: " << it+1 << "/" << numberOfChunks;
        std::flush(std::cout);
        // Calculate image boarders
        xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
        yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

        // Get chunk of our image, mask and result. Apply padding to the result to ensure that the median filter will run correctly.
        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(*mask, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subMask);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

        // Allocate GPU memory for the original image, mask and its result
        CHECK_CUDA(cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize()));
        // Length of columns
        nSrcStep = subImage.cols;

        CHECK_CUDA(cudaMalloc((void **) &deviceMask, subMask.total() * subMask.elemSize()));
        // Length of columns
        nMaskStep = subMask.cols;

        CHECK_CUDA(cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize()));
        // Length of columns
        nResStep = subResult.cols;

        // Copy image from CPU to GPU
        CHECK_CUDA(cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(deviceMask, subMask.data, subMask.total() * subMask.elemSize(), cudaMemcpyHostToDevice));

        // Apply median filter
        subImageDims = {subImage.cols, subImage.rows};
        threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
        numBlocks = dim3(ceil(float(subImageDims.x) / threadsPerBlock.x), ceil(float(subImageDims.y) / threadsPerBlock.y));
        // Run median filter
        medianFilterMaskedKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                                 deviceResult, nResStep,
                                                                 deviceMask, nMaskStep,
                                                                 subImageDims);

        CHECK_CUDA(cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost));

        // Free reserved memory
        CHECK_CUDA(cudaFree(deviceImage));
        CHECK_CUDA(cudaFree(deviceResult));
        CHECK_CUDA(cudaFree(deviceMask));
        CHECK_CUDA(cudaDeviceSynchronize());

        cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, subResult.cols - 2*MEDIAN_KERNEL_SIZE, subResult.rows - 2*MEDIAN_KERNEL_SIZE);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    // Fix output after \r
    std::cout << std::endl;

    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);
    croppedImage = cv::Mat(*mask, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, mask->cols - 2*MEDIAN_KERNEL_SIZE, mask->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*mask);
    return std::make_shared<cv::Mat>(result);
}

cv::Mat PLImg::cuda::raw::CUDAhistogram(const cv::Mat &image, uint *minLabel, uint *maxLabel) {
    uint* deviceImage;
    uint* deviceHistogram;

    CHECK_CUDA(cudaMalloc(&deviceImage, image.total() * sizeof(uint)));
    CHECK_CUDA(cudaMemcpy(deviceImage, image.data, image.total() * sizeof(uint), cudaMemcpyHostToDevice));

    uint* deviceMinLabel = nullptr;
    uint *deviceMaxLabel = nullptr;
    deviceMaxLabel = thrust::max_element(thrust::device, deviceImage, deviceImage + image.total());
    deviceMinLabel = thrust::min_element(thrust::device, deviceImage, deviceImage + image.total());
    CHECK_CUDA(cudaMemcpy(minLabel, deviceMinLabel, sizeof(uint), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(maxLabel, deviceMaxLabel, sizeof(uint), cudaMemcpyDeviceToHost));

    std::cout << "Min Label = " << *minLabel << ", Max Label = " << *maxLabel << std::endl;
    std::cout << (*maxLabel - *minLabel) << std::endl;

    CHECK_CUDA(cudaMalloc(&deviceHistogram, (*maxLabel - *minLabel) * sizeof(uint)));
    CHECK_CUDA(cudaMemset(deviceHistogram, 0, (*maxLabel - *minLabel) * sizeof(uint)));

    dim3 threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
    dim3 numBlocks = dim3(ceil(float(image.cols) / threadsPerBlock.x), ceil(float(image.rows) / threadsPerBlock.y));

    histogram<<<numBlocks, threadsPerBlock>>>(deviceImage, image.cols, image.rows, deviceHistogram, *minLabel, *maxLabel);
    CHECK_CUDA(cudaFree(deviceImage));

    cv::Mat hostHistogram(*maxLabel - *minLabel, 1, CV_32SC1);

    CHECK_CUDA(cudaMemcpy(hostHistogram.data, deviceHistogram, (*maxLabel - *minLabel) * sizeof(uint), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(deviceHistogram));

    return hostHistogram;
}

