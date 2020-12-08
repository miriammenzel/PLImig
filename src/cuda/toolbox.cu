//
// Created by jreuter on 07.12.20.
//

#include "toolbox.cuh"
#include <unistd.h>

__device__ void sortArray(float* array, uint start, uint stop) {
    __shared__ float swap;
    for (uint c = start ; c < stop - 1; c++)
    {
        for (uint d = start ; d < stop - c - 1; d++)
        {
            if (array[d] > array[d+1])
            {
                swap       = array[d];
                array[d]   = array[d+1];
                array[d+1] = swap;
            }
        }
    }
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
                sortArray(buffer, 0, validValues);
                result_image[rx + ry * result_image_stride] = buffer[validValues / 2];
            } else {
                result_image[rx + ry * result_image_stride] = 0;
            }
        } else {
            result_image[rx + ry * result_image_stride] = 0;
        }
    }

}

std::shared_ptr<cv::Mat> PLImg::filters::callCUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask) {
    cv::copyMakeBorder(*image, *image, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(*mask, *mask, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, cv::BORDER_REPLICATE);

    // Error objects
    cudaError_t err;

    float* deviceImage;
    err = cudaMalloc((void**) &deviceImage, image->total() * image->elemSize());
    if(err != cudaSuccess) {
        std::cerr << "Could not allocate enough memory for original transmittance \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    // Length of columns
    ulong nSrcStep = image->cols;

    uchar* deviceMask;
    err = cudaMalloc((void**) &deviceMask, mask->total() * mask->elemSize());
    if(err != cudaSuccess) {
        std::cerr << "Could not allocate enough memory for mask \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    // Length of columns
    ulong nMaskStep = mask->cols;

    float* deviceResult;
    err = cudaMalloc((void**) &deviceResult, image->total() * image->elemSize());
    if(err != cudaSuccess) {
        std::cerr << "Could not allocate enough memory for resulting image \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    // Length of columns
    ulong nResStep = image->cols;

    // Copy image from CPU to GPU
    err = cudaMemcpy(deviceImage, image->data, image->total() * image->elemSize(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        std::cerr << "Could not copy image from host to device \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(deviceMask, mask->data, mask->total() * mask->elemSize(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        std::cerr << "Could not copy mask from host to device \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Apply median filter
    // Set size where median filter will be applied
    int2 roi = {image->cols - 2 * KERNEL_SIZE, image->rows - 2 * KERNEL_SIZE};
    // Median kernel
    int2 anchor = {KERNEL_SIZE / 2, KERNEL_SIZE / 2};
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    int2 pSrcOffset = {KERNEL_SIZE, KERNEL_SIZE};
    int2 pResultOffset = {KERNEL_SIZE, KERNEL_SIZE};
    int2 pMaskOffset = {KERNEL_SIZE, KERNEL_SIZE};

    // Apply median filter
    dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);
    dim3 numBlocks(roi.x / threadsPerBlock.x, roi.y / threadsPerBlock.y);

    // Run median filter
    medianFilterMaskedKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep, pSrcOffset,
                                                             deviceResult, nResStep, pResultOffset,
                                                             deviceMask, nMaskStep, pMaskOffset,
                                                             roi, anchor);

    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    err = cudaMemcpy(result.data, deviceResult, image->total() * image->elemSize(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        std::cerr << "Could not copy image from device to host \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free reserved memory
    cudaFree(deviceImage);
    cudaFree(deviceResult);
    cudaFree(deviceMask);

    // Convert result data to OpenCV image for further calculations
    // Remove padding added at the top of the function
    cv::Mat croppedImage = cv::Mat(result, cv::Rect(KERNEL_SIZE, KERNEL_SIZE, result.cols - 2 * KERNEL_SIZE, result.rows - 2 * KERNEL_SIZE));
    croppedImage.copyTo(result);
    croppedImage = cv::Mat(*image, cv::Rect(KERNEL_SIZE, KERNEL_SIZE, image->cols - 2 * KERNEL_SIZE, image->rows - 2 * KERNEL_SIZE));
    croppedImage.copyTo(*image);
    croppedImage = cv::Mat(*mask, cv::Rect(KERNEL_SIZE, KERNEL_SIZE, mask->cols - 2 * KERNEL_SIZE, mask->rows - 2 * KERNEL_SIZE));
    croppedImage.copyTo(*mask);

    return std::make_shared<cv::Mat>(result);
}


