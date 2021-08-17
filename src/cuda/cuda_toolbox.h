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

#ifndef PLIMG_TOOLBOX_CUH
#define PLIMG_TOOLBOX_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_kernels.h"
#include "define.h"
#include <driver_types.h>
#include <iostream>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <opencv2/opencv.hpp>

/**
 * @file
 * @brief PLImg::cuda::filters functions
 */
namespace PLImg::cuda::raw {
    namespace labeling {
        cv::Mat CUDAConnectedComponents(const cv::Mat& image, uint* maxLabelNumber);
        cv::Mat CUDAConnectedComponentsUF(const cv::Mat& image, uint* maxLabelNumber);
    }

    namespace filters {
        /**
         * @brief CUDAmedianFilter
         * @param image
         * @return
         */
        void CUDAmedianFilter(cv::Mat& image, cv::Mat& result);
        /**
         * @brief CUDAmedianFilterMasked
         * @param image
         * @param mask
         * @return
         */
        void CUDAmedianFilterMasked(cv::Mat& image, cv::Mat& mask, cv::Mat& result);
    }

    cv::Mat CUDAhistogram(const cv::Mat& image, float minLabel, float maxLabel, uint numBins);
}


#endif //PLIMG_TOOLBOX_CUH
