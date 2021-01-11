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

__global__ void medianFilterKernel(const float* image, int image_stride, int2 image_offset,
                                   float* result_image, int result_image_stride, int2 result_offset,
                                   int2 roi, int2 anchor) {
    uint thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint x = thread_x - anchor.x + image_offset.x;
    uint y = thread_y - anchor.y + image_offset.y;
    uint rx = thread_x - anchor.x + result_offset.x;
    uint ry = thread_y - anchor.y + result_offset.y;

    uint validValues = 0;
    int cy_bound;

    float buffer[4 * KERNEL_SIZE * KERNEL_SIZE];

    if(x > KERNEL_SIZE && x < roi.x && y > KERNEL_SIZE && y < roi.y) {
        // Transfer image pixels to our kernel for median filtering application
        for (int cx = -KERNEL_SIZE; cx <= KERNEL_SIZE; ++cx) {
            cy_bound = sqrtf(KERNEL_SIZE * KERNEL_SIZE - cx * cx);
            for (int cy = -cy_bound; cy <= cy_bound; ++cy) {
                buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                ++validValues;
            }
        }
        shellSort(buffer, 0, validValues);
        result_image[rx + ry * result_image_stride] = buffer[validValues / 2];
    } else {
        result_image[rx + ry * result_image_stride] = 0;
    }
    //printf("Valid values: %d\n", validValues);
}

__global__ void medianFilterMaskedKernel(const float* image, int image_stride, int2 image_offset,
                                         float* result_image, int result_image_stride, int2 result_offset,
                                         const uchar* mask, int mask_stride, int2 mask_offset,
                                         int2 roi, int2 anchor) {
    uint thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint x = thread_x - anchor.x + image_offset.x;
    uint y = thread_y - anchor.y + image_offset.y;
    uint rx = thread_x - anchor.x + result_offset.x;
    uint ry = thread_y - anchor.y + result_offset.y;
    uint mx = thread_x - anchor.x + mask_offset.x;
    uint my = thread_y - anchor.y + mask_offset.y;

    float buffer[KERNEL_SIZE * KERNEL_SIZE];
    uint validValues = 0;

    if(x > KERNEL_SIZE && x < roi.x && y > KERNEL_SIZE && y < roi.y) {
        if(mask[mx + my * mask_stride]) {
            // Transfer image pixels to our kernel for median filtering application
            for (uint cx = 0; cx < KERNEL_SIZE; ++cx) {
                for (uint cy = 0; cy < KERNEL_SIZE; ++cy) {
                    if (mask[mx + cx + (my + cy) * mask_stride] != 0) {
                        buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                        ++validValues;
                    }
                }
            }
            if (validValues > 1) {
                shellSort(buffer, 0, validValues);
                result_image[rx + ry * result_image_stride] = buffer[validValues / 2];
            } else if (validValues == 1) {
                result_image[rx + ry * result_image_stride] = buffer[0];
            } else {
                result_image[rx + ry * result_image_stride] = 0;
            }
        } else {
            result_image[rx + ry * result_image_stride] = 0;
        }
    }

}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::callCUDAmedianFilter(const std::shared_ptr<cv::Mat>& image) {
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());

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
    uint chunksPerDim = fmax(1, numberOfChunks/2);

    float* deviceImage, *deviceResult;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nResStep;
    // Apply median filter
    // Set size where median filter will be applied
    int2 roi;
    // Median kernel
    int2 anchor = {KERNEL_SIZE, KERNEL_SIZE};
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    int2 pSrcOffset = {KERNEL_SIZE, KERNEL_SIZE};
    int2 pResultOffset = {KERNEL_SIZE, KERNEL_SIZE};
    dim3 threadsPerBlock, numBlocks;

    cv::Mat subImage, subResult, croppedImage;
    for(uint it = 0; it < numberOfChunks; ++it) {
        // Calculate image boarders
        xMin = (it % chunksPerDim) * image->cols / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * image->cols / chunksPerDim, image->cols);
        yMin = (it / chunksPerDim) * image->rows / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * image->rows / chunksPerDim, image->rows);

        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);

        cv::copyMakeBorder(subImage, subImage, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);
        cv::copyMakeBorder(subResult, subResult, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);

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
        roi = {subImage.cols - 2 * KERNEL_SIZE, subImage.rows - 2 * KERNEL_SIZE};
        threadsPerBlock = dim3(NUM_THREADS, NUM_THREADS);
        numBlocks = dim3(roi.x / threadsPerBlock.x, roi.y / threadsPerBlock.y);
        // Run median filter
        medianFilterKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep, pSrcOffset,
                                                           deviceResult, nResStep, pResultOffset,
                                                           roi, anchor);
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

        cv::Rect srcRect = cv::Rect(KERNEL_SIZE, KERNEL_SIZE, subResult.cols - 2*KERNEL_SIZE, subResult.rows - 2*KERNEL_SIZE);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::callCUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask) {
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());

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
    uint chunksPerDim = fmax(1, numberOfChunks/2);

    float* deviceImage, *deviceResult;
    uchar* deviceMask;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nMaskStep, nResStep;
    // Apply median filter
    // Set size where median filter will be applied
    int2 roi;
    // Median kernel
    int2 anchor = {KERNEL_SIZE / 2, KERNEL_SIZE / 2};
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    int2 pSrcOffset = {KERNEL_SIZE, KERNEL_SIZE};
    int2 pResultOffset = {KERNEL_SIZE, KERNEL_SIZE};
    int2 pMaskOffset = {KERNEL_SIZE, KERNEL_SIZE};
    dim3 threadsPerBlock, numBlocks;

    cv::Mat subImage, subMask, subResult, croppedImage;
    for(uint it = 0; it < numberOfChunks; ++it) {
        // Calculate image boarders
        xMin = (it % chunksPerDim) * image->cols / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * image->cols / chunksPerDim, image->cols);
        yMin = (it / chunksPerDim) * image->rows / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * image->rows / chunksPerDim, image->rows);

        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(*mask, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subMask);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);

        cv::copyMakeBorder(subImage, subImage, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);
        cv::copyMakeBorder(subResult, subResult, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);
        cv::copyMakeBorder(subMask, subMask, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);

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
        roi = {subImage.cols - 2 * KERNEL_SIZE, subImage.rows - 2 * KERNEL_SIZE};
        threadsPerBlock = dim3(NUM_THREADS, NUM_THREADS);
        numBlocks = dim3(roi.x / threadsPerBlock.x, roi.y / threadsPerBlock.y);
        // Run median filter
        medianFilterMaskedKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep, pSrcOffset,
                                                                 deviceResult, nResStep, pResultOffset,
                                                                 deviceMask, nMaskStep, pMaskOffset,
                                                                 roi, anchor);

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

        cv::Rect srcRect = cv::Rect(KERNEL_SIZE, KERNEL_SIZE, subResult.cols - 2*KERNEL_SIZE, subResult.rows - 2*KERNEL_SIZE);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    return std::make_shared<cv::Mat>(result);
}


