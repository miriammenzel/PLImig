/*
    MIT License

    Copyright (c) 2020 Forschungszentrum Jülich / Jan André Reuter.

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

#ifndef PLIMG_TOOLBOX_CUH
#define PLIMG_TOOLBOX_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

/// Number of CUDA Kernel threads used for kernel execution
#define CUDA_KERNEL_NUM_THREADS 32
/// Fixed median kernel size
#define MEDIAN_KERNEL_SIZE 10

/**
 * @file
 * @brief PLImg::cuda::filters functions
 */
namespace PLImg::cuda::filters {
    /**
     * @brief callCUDAmedianFilter
     * @param image
     * @return
     */
    std::shared_ptr<cv::Mat> callCUDAmedianFilter(const std::shared_ptr<cv::Mat>& image);
    /**
     * @brief callCUDAmedianFilterMasked
     * @param image
     * @param mask
     * @return
     */
    std::shared_ptr<cv::Mat> callCUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image,
                                                        const std::shared_ptr<cv::Mat>& mask);
}

#endif //PLIMG_TOOLBOX_CUH
