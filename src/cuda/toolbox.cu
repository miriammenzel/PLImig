//
// Created by jreuter on 07.12.20.
//

#include "toolbox.cuh"
#include <chrono>

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
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint validValues = 0;
    int cy_bound;

    float buffer[4 * MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];

    if(x >= MEDIAN_KERNEL_SIZE && x < imageDims.x - MEDIAN_KERNEL_SIZE && y >= MEDIAN_KERNEL_SIZE && y < imageDims.y - MEDIAN_KERNEL_SIZE) {
        // Transfer image pixels to our kernel for median filtering application
        for (int cx = -MEDIAN_KERNEL_SIZE; cx <= MEDIAN_KERNEL_SIZE; ++cx) {
            cy_bound = sqrtf(MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE - cx * cx);
            for (int cy = -cy_bound; cy <= cy_bound; ++cy) {
                buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                ++validValues;
            }
        }
        shellSort(buffer, 0, validValues);
        result_image[x + y * result_image_stride] = buffer[validValues / 2];
    } else {
        result_image[x + y * result_image_stride] = 0.0f;
    }
}

__global__ void medianFilterMaskedKernel(const float* image, int image_stride,
                                         float* result_image, int result_image_stride,
                                         const uchar* mask, int mask_stride,
                                         int2 imageDims) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint validValues = 0;
    int cy_bound;

    float buffer[4 * MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];

    if(x >= MEDIAN_KERNEL_SIZE && x < imageDims.x - MEDIAN_KERNEL_SIZE && y >= MEDIAN_KERNEL_SIZE && y < imageDims.y - MEDIAN_KERNEL_SIZE) {
        if(mask[x + y * mask_stride]) {
            // Transfer image pixels to our kernel for median filtering application
            for (int cx = -MEDIAN_KERNEL_SIZE; cx < MEDIAN_KERNEL_SIZE; ++cx) {
                cy_bound = sqrtf(MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE - cx * cx);
                for (int cy = -cy_bound; cy < cy_bound; ++cy) {
                    if (mask[x + cx + (y + cy) * mask_stride] != 0) {
                        buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                        ++validValues;
                    }
                }
            }
            if (validValues > 1) {
                shellSort(buffer, 0, validValues);
                result_image[x + y * result_image_stride] = buffer[validValues / 2];
            } else if (validValues == 1) {
                result_image[x + y * result_image_stride] = buffer[0];
            } else {
                result_image[x + y * result_image_stride] = 0;
            }
        } else {
            result_image[x + y * result_image_stride] = 0;
        }
    }

}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::callCUDAmedianFilter(const std::shared_ptr<cv::Mat>& image) {
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // Error objects
    cudaError_t err;

    uint numberOfChunks = 1;
    ulong freeMem;
    err = cudaMemGetInfo(&freeMem, nullptr);
    if(err != cudaSuccess) {
        std::cerr << "Could not get free memory! \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    if(double(image->total()) * image->elemSize() * 2.1 > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(image->total() * image->elemSize() * 2.1 / double(freeMem)) / log(4))));
    }
    uint chunksPerDim = fmax(1, sqrtf(numberOfChunks));

    float* deviceImage, *deviceResult;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;

    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    int2 subImageDims;

    cv::Mat subImage, subResult, croppedImage;
    for(uint it = 0; it < numberOfChunks; ++it) {
        // Calculate image boarders
        xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
        yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

        err = cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for original transmittance \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nSrcStep = subImage.cols;

        err = cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for resulting image \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nResStep = subResult.cols;

        // Copy image from CPU to GPU
        err = cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Apply median filter
        auto start = std::chrono::high_resolution_clock::now();
        subImageDims = {subImage.cols, subImage.rows};
        threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
        numBlocks = dim3(subImageDims.x / threadsPerBlock.x, subImageDims.y / threadsPerBlock.y);
        // Run median filter
        medianFilterKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                           deviceResult, nResStep,
                                                           subImageDims);
        cudaDeviceSynchronize();
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";

        err = cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from device to host \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Free reserved memory
        cudaFree(deviceImage);
        cudaFree(deviceResult);     

        cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, subResult.cols - 2*MEDIAN_KERNEL_SIZE, subResult.rows - 2*MEDIAN_KERNEL_SIZE);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);
    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::callCUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask) {
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(*mask, *mask, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // Error objects
    cudaError_t err;

    uint numberOfChunks = 1;
    ulong freeMem;
    err = cudaMemGetInfo(&freeMem, nullptr);
    if(err != cudaSuccess) {
        std::cerr << "Could not get free memory! \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    if(double(image->total()) * image->elemSize() * 3.1 > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(image->total() * image->elemSize() * 3.1 / double(freeMem)) / log(4))));
    }
    uint chunksPerDim = fmax(1, sqrtf(numberOfChunks));

    float* deviceImage, *deviceResult;
    uchar* deviceMask;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nMaskStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;

    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    int2 subImageDims;

    cv::Mat subImage, subMask, subResult, croppedImage;
    for(uint it = 0; it < numberOfChunks; ++it) {
        // Calculate image boarders
        xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
        yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(*mask, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subMask);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

        err = cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for original transmittance \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nSrcStep = subImage.cols;

        err = cudaMalloc((void **) &deviceMask, subMask.total() * subMask.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for mask \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nMaskStep = subMask.cols;

        err = cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for resulting image \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nResStep = subResult.cols;

        // Copy image from CPU to GPU
        err = cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(deviceMask, subMask.data, subMask.total() * subMask.elemSize(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy mask from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Apply median filter
        subImageDims = {subImage.cols, subImage.rows};
        threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
        numBlocks = dim3(subImageDims.x / threadsPerBlock.x, subImageDims.y / threadsPerBlock.y);
        // Run median filter
        medianFilterMaskedKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                                 deviceResult, nResStep,
                                                                 deviceMask, nMaskStep,
                                                                 subImageDims);

        err = cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from device to host \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Free reserved memory
        cudaFree(deviceImage);
        cudaFree(deviceResult);
        cudaFree(deviceMask);

        cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, subResult.cols - 2*MEDIAN_KERNEL_SIZE, subResult.rows - 2*MEDIAN_KERNEL_SIZE);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);
    croppedImage = cv::Mat(*mask, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, mask->cols - 2*MEDIAN_KERNEL_SIZE, mask->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*mask);
    return std::make_shared<cv::Mat>(result);
}


